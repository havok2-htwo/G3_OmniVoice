import React from "react";
import ReactDOM from "react-dom/client";

import DemoApp from "./demo/DemoApp";
import "./styles/app.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <DemoApp />
  </React.StrictMode>,
);
