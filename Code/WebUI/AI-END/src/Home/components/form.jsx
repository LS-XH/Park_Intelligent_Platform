import { Tabs, ConfigProvider } from "antd";
import Loginform from "./loginForm";
import RegisterForm from "./registerForm";


const Form = () => {
    const items = [
        {
            key: '1',
            label: `登录`,
            children: <Loginform />,
        },
        {
            key: '2',
            label: `注册`,
            children: <RegisterForm />,
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
                            colorPrimary: '#1890ff',
                            inkBarColor: "#1890ff"
                        }
                    }
                }}
            >
                <Tabs
                    items={items}
                    centered
                    style={{ flex: 1, display: 'flex', flexDirection: 'column' }}
                />
            </ConfigProvider>
        </div>
    );
};

export default Form;