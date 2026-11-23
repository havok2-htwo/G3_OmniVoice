import React from "react";
import ReactDOM from "react-dom/client";

import { LandingPage } from "./landing/LandingPage";
import "./g3-base.css";
import "./styles/app.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <LandingPage />
  </React.StrictMode>,
);
