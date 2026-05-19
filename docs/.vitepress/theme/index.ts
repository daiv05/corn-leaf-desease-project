import DefaultTheme from "vitepress/theme";
import ImageCarousel from "../components/ImageCarousel.vue";

export default {
  ...DefaultTheme,
  enhanceApp({ app }) {
    app.component("ImageCarousel", ImageCarousel);
  },
};
