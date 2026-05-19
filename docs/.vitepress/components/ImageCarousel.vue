<template>
  <div class="carousel">
    <div class="carousel-main">
      <button class="carousel-nav" type="button" @click="prev" :disabled="items.length === 0">
        ‹
      </button>
      <button class="carousel-image" type="button" @click="openZoom" :disabled="!current.src">
        <img :src="current.src" :alt="current.alt" loading="lazy" />
      </button>
      <button class="carousel-nav" type="button" @click="next" :disabled="items.length === 0">
        ›
      </button>
    </div>

    <div class="carousel-actions">
      <span class="carousel-caption">{{ current.alt }}</span>
      <div class="carousel-buttons">
        <button type="button" @click="openZoom" :disabled="!current.src">Zoom</button>
        <a class="download" :href="current.src" :download="downloadName(current.src)" :aria-disabled="!current.src">
          Descargar
        </a>
      </div>
    </div>

    <div class="carousel-thumbs">
      <button
        v-for="(img, i) in items"
        :key="img.src + i"
        type="button"
        class="thumb"
        :class="{ active: i === index }"
        @click="select(i)"
      >
        <img :src="img.src" :alt="img.alt" loading="lazy" />
      </button>
    </div>

    <div v-if="zoomOpen" class="zoom-overlay" @click.self="closeZoom">
      <div class="zoom-panel">
        <div class="zoom-toolbar">
          <button type="button" @click="zoomOut">-</button>
          <input v-model.number="zoom" type="range" min="1" max="3" step="0.1" />
          <button type="button" @click="zoomIn">+</button>
          <button type="button" @click="closeZoom">Cerrar</button>
        </div>
        <div class="zoom-stage">
          <img :src="current.src" :alt="current.alt" :style="{ transform: `scale(${zoom})` }" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'

const props = defineProps({
  images: {
    type: Array,
    required: true,
  },
})

const index = ref(0)
const zoomOpen = ref(false)
const zoom = ref(1)

const items = computed(() =>
  props.images.map((img, i) => {
    if (typeof img === 'string') {
      return { src: img, alt: `Imagen ${i + 1}` }
    }

    return {
      src: img.src,
      alt: img.alt || `Imagen ${i + 1}`,
    }
  })
)

const current = computed(() => items.value[index.value] || { src: '', alt: '' })

function prev() {
  if (!items.value.length) return
  index.value = (index.value - 1 + items.value.length) % items.value.length
}

function next() {
  if (!items.value.length) return
  index.value = (index.value + 1) % items.value.length
}

function select(i) {
  index.value = i
}

function openZoom() {
  zoomOpen.value = true
}

function closeZoom() {
  zoomOpen.value = false
  zoom.value = 1
}

function zoomIn() {
  zoom.value = Math.min(3, Number((zoom.value + 0.2).toFixed(1)))
}

function zoomOut() {
  zoom.value = Math.max(1, Number((zoom.value - 0.2).toFixed(1)))
}

function downloadName(src) {
  if (!src) return 'imagen.jpg'
  const parts = src.split('/')
  return parts[parts.length - 1] || 'imagen.jpg'
}

watch(
  () => props.images,
  () => {
    if (index.value >= items.value.length) index.value = 0
  }
)
</script>

<style scoped>
.carousel {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin: 16px 0 24px;
  padding: 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  background: linear-gradient(135deg, rgba(18, 18, 18, 0.04), rgba(18, 18, 18, 0));
}

.carousel-main {
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  gap: 12px;
}

.carousel-nav {
  height: 44px;
  width: 44px;
  border-radius: 50%;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  font-size: 22px;
}

.carousel-image {
  padding: 0;
  border: 1px solid var(--vp-c-divider);
  border-radius: 14px;
  overflow: hidden;
  background: #0f1115;
}

.carousel-image img {
  display: block;
  width: 100%;
  max-height: 420px;
  object-fit: contain;
}

.carousel-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.carousel-caption {
  color: var(--vp-c-text-2);
}

.carousel-buttons {
  display: flex;
  gap: 8px;
  align-items: center;
}

.carousel-buttons button,
.carousel-buttons .download {
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
  padding: 6px 10px;
  font-size: 14px;
  color: var(--vp-c-text-1);
}

.carousel-buttons .download[aria-disabled='true'] {
  pointer-events: none;
  opacity: 0.5;
}

.carousel-thumbs {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(72px, 1fr));
  gap: 8px;
}

.thumb {
  border: 2px solid transparent;
  border-radius: 10px;
  padding: 0;
  overflow: hidden;
  background: #0f1115;
}

.thumb.active {
  border-color: #38b000;
}

.thumb img {
  width: 100%;
  height: 60px;
  object-fit: cover;
  display: block;
}

.zoom-overlay {
  position: fixed;
  inset: 0;
  background: rgba(10, 12, 16, 0.75);
  display: grid;
  place-items: center;
  z-index: 50;
  padding: 16px;
}

.zoom-panel {
  max-width: min(1100px, 95vw);
  width: 100%;
  background: #0f1115;
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  overflow: hidden;
}

.zoom-toolbar {
  display: flex;
  gap: 8px;
  align-items: center;
  padding: 10px;
  background: rgba(255, 255, 255, 0.04);
}

.zoom-toolbar button {
  border: 1px solid rgba(255, 255, 255, 0.15);
  background: transparent;
  color: #f8f9fb;
  border-radius: 6px;
  padding: 4px 8px;
}

.zoom-toolbar input[type='range'] {
  flex: 1;
}

.zoom-stage {
  display: grid;
  place-items: center;
  max-height: 70vh;
  padding: 16px;
  overflow: auto;
}

.zoom-stage img {
  max-width: 100%;
  max-height: 70vh;
  transform-origin: center center;
}

@media (max-width: 720px) {
  .carousel-main {
    grid-template-columns: auto 1fr auto;
  }

  .carousel-image img {
    max-height: 300px;
  }
}
</style>
