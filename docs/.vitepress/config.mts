import { defineConfig } from "vitepress";

const esDatasetSidebar = [
  {
    text: "Datasets",
    items: [
      { text: "Recopilación", link: "/es/datasets/" },
      {
        text: "Maize in Field Dataset",
        link: "/es/datasets/maize-in-field-dataset",
      },
      { text: "Maize Diseases", link: "/es/datasets/maize-diseases" },
      { text: "Corn Leaf Diseases", link: "/es/datasets/corn-leaf-diseases" },
      {
        text: "CropDG Unified Multi-Domain",
        link: "/es/datasets/cropdg-unified-multidomain",
      },
      {
        text: "Maize, Beans & Tomatoes (África)",
        link: "/es/datasets/maize-beans-tomatoes-africa",
      },
    ],
  },
  {
    text: "Análisis Exploratorio",
    items: [
      {
        text: "Explorando",
        link: "/es/exploratory-data-analysis/",
      },
    ],
  }
];

// const enDatasetSidebar = [
//   {
//     text: 'Datasets',
//     items: [
//       { text: 'Overview', link: '/en/datasets/' },
//       { text: 'Maize in Field Dataset', link: '/en/datasets/maize-in-field-dataset' },
//       { text: 'Maize Diseases', link: '/en/datasets/maize-diseases' },
//       { text: 'Corn Leaf Diseases', link: '/en/datasets/corn-leaf-diseases' },
//       { text: 'CropDG Unified Multi-Domain', link: '/en/datasets/cropdg-unified-multidomain' },
//       { text: 'Maize, Beans & Tomatoes (Africa)', link: '/en/datasets/maize-beans-tomatoes-africa' },
//     ],
//   },
// ]

export default defineConfig({
  vite: {
    publicDir: "../public",
  },

  head: [["meta", { name: "robots", content: "noindex, nofollow" }]],

  markdown: {
    math: true,
  },

  locales: {
    es: {
      label: "Español",
      lang: "es-SV",
      link: "/es/",
      title: "DoctorMaiz",
      head: [
        [
          "meta",
          {
            name: "description",
            content:
              "Detección de Enfermedades Foliares en Cultivos de Maíz mediante Deep Learning en Dispositivos Móviles",
          },
        ],
        [
          "meta",
          {
            name: "keywords",
            content:
              "maíz, enfermedades foliares, detección, deep learning, dispositivos móviles, datasets, agricultura de precisión",
          },
        ],
        ["link", { rel: "icon", type: "image/svg+xml", href: "/logo.svg" }],
        ["link", { rel: "icon", type: "image/png", href: "/logo.png" }],
      ],
      description:
        "Detección de Enfermedades Foliares en Cultivos de Maíz mediante Deep Learning en Dispositivos Móviles",
      themeConfig: {
        nav: [
          { text: "Inicio", link: "/es/" },
          { text: "Datasets", link: "/es/datasets/" },
          { text: "Análisis Exploratorio", link: "/es/exploratory-data-analysis/" },
        ],
        sidebar: {
          "/es/": esDatasetSidebar,
          "/es/datasets/": esDatasetSidebar,
        },
        outlineTitle: "En esta página",
        docFooter: { prev: "Anterior", next: "Siguiente" },
        darkModeSwitchLabel: "Apariencia",
        sidebarMenuLabel: "Menú",
        returnToTopLabel: "Volver arriba",
        langMenuLabel: "Idioma",
        logo: "/logo.svg",
      },
    },
    // en: {
    //   label: 'English',
    //   lang: 'en',
    //   link: '/en/',
    //   title: 'Corn Leaf Disease Detection',
    //   description: 'Detection of Corn Leaf Diseases using Deep Learning on Mobile Devices',
    //   themeConfig: {
    //     nav: [
    //       { text: 'Home', link: '/en/' },
    //       { text: 'Datasets', link: '/en/datasets/' },
    //     ],
    //     sidebar: {
    //       '/en/': enDatasetSidebar,
    //       '/en/datasets/': enDatasetSidebar,
    //     },
    //   },
    // },
  },

  themeConfig: {
    search: {
      provider: "local",
    },
  },
  
});
