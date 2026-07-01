"""Wrapper delgado sobre la CLI `vastai` (ya autenticada por el usuario) para orquestar
entrenamiento en GPU remota de forma reproducible. No reemplaza a `vastai`, solo encadena
los comandos típicos del flujo: buscar oferta -> crear instancia -> correr comandos make ->
traer resultados -> destruir instancia.

Requiere: `pip install vastai` y `vastai set api-key <tu-api-key>` ya configurado.
"""

import argparse
import re
import subprocess
import sys

REMOTE_WORKSPACE = "/workspace/corn-leaf-desease-project"
REMOTE_DATASET_ROOT = "/workspace/data"
# Necesita Python >= 3.11 (ver pyproject.toml: requires-python). Las imágenes oficiales
# "pytorch/pytorch:*-runtime" traen Python 3.10 vía conda y rompen la instalación del
# paquete; la plantilla vastai/pytorch en cambio sí trae Python 3.12.
DEFAULT_IMAGE = "vastai/pytorch:2.6.0-cuda-12.6.3-py312"
ONSTART_SCRIPT = "scripts/vastai/onstart.sh"


def _run(cmd: list[str], dry_run: bool) -> subprocess.CompletedProcess | None:
    print(f"$ {' '.join(cmd)}")
    if dry_run:
        return None
    return subprocess.run(cmd, check=True)


def _get_ssh_target(instance_id: str) -> tuple[str, str, str]:
    result = subprocess.run(
        ["vastai", "ssh-url", str(instance_id)], capture_output=True, text=True, check=True
    )
    url = result.stdout.strip()
    match = re.match(r"ssh://([^@]+)@([^:]+):(\d+)", url)
    if not match:
        raise SystemExit(f"No se pudo interpretar la URL ssh de vast.ai: {url!r}")
    return match.group(1), match.group(2), match.group(3)


def cmd_search(args: argparse.Namespace) -> None:
    _run(["vastai", "search", "offers", *args.query], dry_run=args.dry_run)


def cmd_create(args: argparse.Namespace) -> None:
    cmd = [
        "vastai",
        "create",
        "instance",
        str(args.offer_id),
        "--image",
        args.image,
        "--disk",
        str(args.disk),
        "--ssh",
        "--direct",
        "--onstart",
        ONSTART_SCRIPT,
    ]
    if args.env:
        for key_value in args.env:
            cmd += ["--env", f"-e {key_value}"]
    _run(cmd, dry_run=args.dry_run)


def cmd_run(args: argparse.Namespace) -> None:
    if args.dry_run:
        print(f"$ ssh <{args.instance_id}> 'cd {REMOTE_WORKSPACE} && {' '.join(args.command)}'")
        return
    user, host, port = _get_ssh_target(args.instance_id)
    remote_cmd = f"cd {REMOTE_WORKSPACE} && {' '.join(args.command)}"
    subprocess.run(["ssh", "-p", port, f"{user}@{host}", remote_cmd], check=True)


def cmd_sync(args: argparse.Namespace) -> None:
    remote_path = f"{args.instance_id}:{REMOTE_DATASET_ROOT}/{args.remote_subpath}"
    _run(["vastai", "copy", remote_path, args.local_path], dry_run=args.dry_run)


def cmd_destroy(args: argparse.Namespace) -> None:
    _run(["vastai", "destroy", "instance", str(args.instance_id)], dry_run=args.dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Imprime los comandos sin ejecutarlos."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    p_search = subparsers.add_parser("search", help="Busca ofertas de GPU (vastai search offers).")
    p_search.add_argument(
        "query", nargs=argparse.REMAINDER, help="Filtros, p.ej. 'gpu_name=RTX_3090 num_gpus=1'"
    )
    p_search.set_defaults(func=cmd_search)

    p_create = subparsers.add_parser("create", help="Crea una instancia a partir de una oferta.")
    p_create.add_argument("offer_id")
    p_create.add_argument("--image", default=DEFAULT_IMAGE)
    p_create.add_argument("--disk", type=int, default=32, help="GB de disco.")
    p_create.add_argument(
        "--env",
        action="append",
        default=None,
        help="Variables a pasar al onstart, formato KEY=VALUE (repetible), "
        "p.ej. --env HF_DATASET_REPO=usuario/corn-leaf-clean --env HF_TOKEN=hf_xxx",
    )
    p_create.set_defaults(func=cmd_create)

    p_run = subparsers.add_parser(
        "run", help="Corre un comando remoto por ssh (p.ej. targets de Makefile)."
    )
    p_run.add_argument("instance_id")
    p_run.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="p.ej. make splits-baseline && make train-baselines",
    )
    p_run.set_defaults(func=cmd_run)

    p_sync = subparsers.add_parser(
        "sync", help="Copia resultados desde la instancia al equipo local."
    )
    p_sync.add_argument("instance_id")
    p_sync.add_argument(
        "--remote-subpath", default="results", help="Subruta bajo DATASET_ROOT remoto."
    )
    p_sync.add_argument("--local-path", default="./results-remote")
    p_sync.set_defaults(func=cmd_sync)

    p_destroy = subparsers.add_parser("destroy", help="Destruye la instancia (deja de cobrar).")
    p_destroy.add_argument("instance_id")
    p_destroy.set_defaults(func=cmd_destroy)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
