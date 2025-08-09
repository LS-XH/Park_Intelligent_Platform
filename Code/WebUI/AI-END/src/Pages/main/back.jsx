import Board from "./component/gauge";
import LineChart from "./component/lineChart";
const Back = () => {

    return (
        <div
            style={{
                backgroundColor: "black",
            }}
        >
            <Board />
            <LineChart />
        </div>

    );
};

export default Back;