import DefaultTheme from "vitepress/theme";
import ImageCarousel from "../components/ImageCarousel.vue";
import HomeLayout from "./HomeLayout.vue";
import "./custom.css";

export default {
  ...DefaultTheme,
  Layout: HomeLayout,
  enhanceApp({ app }: { app: import("vue").App }) {
    app.component("ImageCarousel", ImageCarousel);
  },
};
