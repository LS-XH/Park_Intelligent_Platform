import { createBrowserRouter } from "react-router-dom";
import Home from "../Home/home";

export const router = createBrowserRouter([
    {
        path: "/",
        children: [
            {
                index: true,
                element: <Home />
            },
        ],
    },
]);