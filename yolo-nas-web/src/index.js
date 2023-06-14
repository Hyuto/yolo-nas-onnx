import React, { useState, Suspense, lazy } from "react";
import ReactDOM from "react-dom/client";
import Loader from "./components/loader";
import "./style/index.css";

const onDev = process.env.NODE_ENV === "development";
const App = lazy(() => import("./App"));

const Main = () => {
  const [show, setShow] = useState(onDev);

  return (
    <>
      {!show && (
        <div className="agreement-con">
          <div className="agreement">
            <h2>⚠️ YOLO-NAS-WEB : Big size webapp!</h2>
            <p>
              Need big amount of data to load application and YOLO-NAS model. Roughly about{" "}
              <strong>61.3 MB</strong> data to load this app, Do you want to proceed?
            </p>
          </div>
          <button
            onClick={() => {
              setShow(true);
            }}
          >
            Yes, I agree
          </button>
        </div>
      )}
      <hr />
      {show && (
        <Suspense fallback={<Loader>Load main component...</Loader>}>
          <App />
        </Suspense>
      )}
    </>
  );
};

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <Main />
  </React.StrictMode>
);
