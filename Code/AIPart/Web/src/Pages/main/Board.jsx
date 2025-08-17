import Gauge from "./component/gauge";
import LineChart from "./component/lineChart";
import { useWebSocket } from "../../Hook/wshook";
import WeatherCard from "./component/weatherCard";
const Board = () => {
    return (
        <div
            style={
                {
                    width: '100%',
                    height: '100%',
                }
            }>
            <Gauge />
            <LineChart />
            <WeatherCard />
        </div>

    );
};

export default Board;