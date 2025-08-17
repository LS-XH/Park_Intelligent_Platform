import { Slider } from "antd";

export const SliderComponent = ({ min, max, currentData, handleChange, type, setCurrentData }) => {

    return (
        <div>
            <Slider
                min={min}
                max={max}
                value={currentData}
                trackStyle={{
                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                    height: 4
                }}
                railStyle={{
                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                    height: 4
                }}

                onChange={(value) => handleChange(type, value)}
            />
        </div>
    );
};