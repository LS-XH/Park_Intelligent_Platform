import { createBrowserRouter } from "react-router-dom";
import Home from "../Home/home";
import Board from "../Pages/main/Board";
import App from "../App";
import Unity from "../unity";
import HeatmapChart from "../Pages/main/component/heatmap";
import NavigationBar from "../Pages/main/component/nav";
import ControlButtons from "../Pages/main/component/button";
import SpeedControl from "../Pages/main/component/contorl";


export const router = createBrowserRouter([
    {
        path: "/",
        children: [
            {
                index: true,
                element: <Home />
            },
            {
                path: "/app",
                element: <App />,
                children: [
                    {
                        //仪表盘和折线图  
                        path: "board",
                        element: <Board />
                    },
                ],

            },
            {
                path: "/unity",
                element: <Unity />,
            }
        ],
    },
]);