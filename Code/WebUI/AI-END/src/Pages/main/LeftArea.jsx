import ControlButtons from "./component/button";
import SpeedControl from "./component/contorl";
import DensityMap from "./component/heatmap";
import RouteSearch from "./component/nav";


const LeftArea = () => {
    return (
        <div
            style={{
                width: "100%",
                height: "100%",
            }}>
            <SpeedControl />
            <RouteSearch />
            <ControlButtons />
            <DensityMap />
        </div>
    );
};

export default LeftArea;