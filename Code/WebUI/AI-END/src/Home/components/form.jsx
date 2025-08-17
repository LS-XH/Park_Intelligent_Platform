import { Tabs, ConfigProvider } from "antd";
import Loginform from "./loginForm";
import RegisterForm from "./registerForm";
import { useState } from "react";


const Form = () => {

    const [activeKey, setActiveKey] = useState('1'); // 添加状态管理

    const handleTabChange = (key) => {
        setActiveKey(key);
    };
    const items = [
        {
            key: '1',
            label: `登录`,
            children: <Loginform />,
        },
        {
            key: '2',
            label: `注册`,
            children: <RegisterForm onTabChange={handleTabChange} />,
        },

    ]

    return (
        <div
            style={{
                background: 'white',
                borderRadius: '10px',
                padding: '20px',
                height: 'auto', // 改为自动高度
                minHeight: '400px', // 设置最小高度而不是固定高度
                display: 'flex',
                flexDirection: 'column',
                boxShadow: '0 8px 16px rgb(0,0,0,0.1)',
                width: '400px',
            }}>
            <ConfigProvider
                theme={{
                    components: {
                        Tabs: {
                            colorPrimary: '#121212',
                            inkBarColor: "#121212"
                        }
                    }
                }}
            >
                <Tabs
                    activeKey={activeKey} // 绑定当前激活的 tab
                    onChange={handleTabChange} // tab 切换时的回调
                    items={items}
                    centered
                    style={{ flex: 1, display: 'flex', flexDirection: 'column' }}
                />
            </ConfigProvider>
        </div>
    );
};

export default Form;         