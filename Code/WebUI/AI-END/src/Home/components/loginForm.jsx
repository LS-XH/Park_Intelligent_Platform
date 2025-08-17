import { Form, Input, Button, Typography, Space, message } from "antd";
import { useState } from "react";
// import '@ant-design/v5-patch-for-react-19';
import { encryptWithAESAndRSA, decryptWithAESAndRSA } from "../../../util/entrypt";
import { login, findPassword, getCodeAPI } from "../../api/service/registerService";
import { useNavigate } from "react-router-dom";

const Loginform = () => {
    const [form] = Form.useForm();
    const { Text } = Typography;
    const [isForgotPassword, setIsForgotPassword] = useState(false); // 切换登录和找回密码
    const [countdown, setCountdown] = useState(0);
    const navigate = useNavigate();
    const handleForgotPassword = () => {
        setIsForgotPassword(true);
    };
    const handleBackToLogin = () => {
        setIsForgotPassword(false);
    };

    const publicKey = `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAsQBfHrU7NYVB8l0kmD79ayRbS2Nmu0gOKIg177flG/MiZd5TIYuH+eOINrFFgu6K1jmTqeUDw5Lm2SPofC1fV++V6yhJu8Vveaa0WhFElSrp5F4vsZ34HB7kpZmH6Vp/u9tdohDrXe+cVdO74ILxsw9CLpEpFrFHmgThVSKtNfwCExZeOT5lN6UKgsxp+HIFbhKWF9NMpmeYw5ie10YevN9Fq9x11aeg+ZgKct1GzF9RfOcX0h6Mz4xu45q5bWRQS+djvprBS5tvYOCVZj9KEanltbFFq71PmiQLdkH7imCFtwHPZzK5TAYeknH+raSjlGDMsijs+I8tR8XpuQcXtwIDAQAB
-----END PUBLIC KEY-----`;
    const privateKey = `-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC1AZj7Ik201dBfzZ9eP3rJCsdt9Hy7RP9KR+xTWnI+VGrW+fDUAzYyFzegXemaXpgePTchdbeioow+JkUPkWQWOmOlmgg+aXMymqgvkix1e3MrepRJkRNVXJJ5KvSJt7OPR5K5B8QTyLXEP8sOLaVpfTJxOmrbe5EwLf1iF63JmL+JoM3vxFZX/kroirb9fYCsWfaZG8IIil5SXlgS7UPd5mpy+pXn0QrhwP2Gw23nG8pX+AwTuL65dJC8DSZMfvFdWZeTQnw6AVS8StMmvXazMNWgOzA65kCJlQB62T69/CYttAjGGXzbWZ6Koe0/TOtpADIdLDOPIe++lxm6Rrn9AgMBAAECggEASPwDbOPIlG2Qb0jQhWawQkc74cyuzK4GCDQXCQcTwKk2SUePwVUoMatl7R5g9rNEwBCr3ayDJqtHRDoXJ69WvZW+n0QMJepMHm/49/GHRrnH1xS+nSlHs+g3UW8uGie92byg30XP3LBWBnM4k5d5Np9aSwiklKpvAQ/SNw7YLsxgG3tjmMgzbzQnNtTdW81BV+A4KIaYVUZnoSEVSzHN3T7WZgG9TLiCm7AowQ8qFTK0Sxu0kO3JGc6G3GkQmR77J4YDIv8O+Da+ITmyVEwxtzuIKNa/VDCtV3Anxit+Hk0xBNsT9Vvdv3upMyjOggjXnWbQYUXN4zbv1IoWlwqOiwKBgQDlzIUZHQPY5t1BWOOXr1w7KRVfxtqm9fKy1x4v+T7f1SPOmFpmAg3Qfv6c1dpON8d6NJS01kuymFW/iPVdvjpoGWqYBSEn0mbz9iBU9quNaU4WhqmlU8OdfqCch0jOK/l+3WCl4BshShpis2mvXPa8HtygkRqQfMVGML2nGttb+wKBgQDJpOTiJSRlcRgj27qXsQjXwrJsEAoZ0cA/UaUlWed/qHBsLE9yk+Y3bxlFc8Sf0wLJejaaCOQ4IS6e81bl7AOT/VBqU77zzBS1uZ4L1dMlJipELgtQcsv06CblDhnRJLAAJ0/xtX6HUqu2v8pGqaLyqsESZXn8TdwxyBKfbnioZwKBgQCIgQ7nNhcc9zajJLw9VIvDEMqDlEo6N4sttR9XfAVfTOryRAoe4kV2fpmcbGQ7ZmL2MtnK+ikJM/hryF2IjAGB6Ocq2pExaIiDjsbx8X1CiTU7qE6JyNJAcgHSOYKEBhc0xygsII29HpnB27WB2AUxBlwkfU18WsGMylM+OnPnlQKBgQC7MtQyhlzVuDq6/4Co1vfopp3R6MoX0jxyDDAPDvn177//DNvs+RVfHUsOyT0fS1xpA4axVdPZsCSB+FMSPRvNRfxj2b+Kwknvs5TgU/AjqtzOUxi55PkoMmX5fC/HlBG48sYrFV2T79HuZPs6wr2+H3wCwiaPbxEfPijbzklBvQKBgBFCAippRBOSX6gol9VJSwzB61Ak9U2vYKucWia4GrEtS/faEUmNKj220qksTEjiACnbTrWojZFKEWx30s+mrkwXdXmNbuq2so5fEvjGQ8rKXCcJNp5/pInPMvhCvw4tUuJ9lEH8EXkDFFRlTVoUnlWofHdTcQreOs/tkHlfrC5M
-----END PRIVATE KEY-----`
    const handleLogin = async (values) => {
        const { encryptedData, encryptedKey } = encryptWithAESAndRSA(
            JSON.stringify({ email: values.email, password: values.password }),
            publicKey
        );
        const response = await login(encryptedData, encryptedKey)


        if (response.data.code === 200) {
            console.log("登录成功", response.data.data.encryptedData);

            navigate("/app/board");
            // //对后台返回的数据进行解密
            const decryptedData = decryptWithAESAndRSA(
                response.data.data.encryptedData,
                response.data.data.encryptedKey,
                privateKey
            );
            console.log("解密后的数据:", JSON.parse(decryptedData));

            // dispatch(setUser(JSON.parse(decryptedData)));
        }
    };
    //获取验证码
    const handleGetCode = async () => {

        const email = form.getFieldValue("email");

        if (!email) {
            message.error("请输入邮箱");
            return;
        }
        console.log("第一步", email);

        // 使用 encryptWithAESAndRSA 函数加密邮箱
        const { encryptedData, encryptedKey } = encryptWithAESAndRSA(JSON.stringify({ email: email }), publicKey);
        const response = await getCodeAPI(encryptedData, encryptedKey);

        if (response.status === 200) {
            setCountdown(60);
            message.success("验证码已发送");
        } else {
            message.error("验证码发送失败");
        }
    };

    //找回密码
    const handlefindPassword = async () => {
        const email = form.getFieldValue("email");
        const code = form.getFieldValue("code");
        const password = form.getFieldValue("password");

        const { encryptedData, encryptedKey } = encryptWithAESAndRSA(
            JSON.stringify({
                users: { email: email, password: password },
                code: code
            }
            ),
            publicKey
        );
        const response = await findPassword(encryptedData, encryptedKey);
        if (response.code === 200) {
            message.success("密码修改成功");
            setIsForgotPassword(false);
        } else {
            message.error("验证码错误");
        }
    };

    if (!isForgotPassword) {
        return (
            <div className="login-form">
                <Form
                    form={form}
                    name="login"
                    onFinish={handleLogin}
                    layout="vertical"
                    autoComplete="off"
                >
                    <Form.Item
                        label="邮箱"
                        name="email"
                        rules={[
                            { required: true, message: "请输入邮箱!" },
                            { type: "email", message: "请输入有效的邮箱!" },
                        ]}
                    >
                        <Input placeholder="请输入邮箱" />
                    </Form.Item>

                    <Form.Item
                        label="密码"
                        name="password"
                        rules={[
                            { required: true, message: "请输入密码!" },
                            { min: 6, message: "密码至少6个字符!" },
                        ]}
                    >
                        <Input.Password placeholder="请输入密码" />
                    </Form.Item>

                    <Form.Item>
                        <Button type="primary" htmlType="submit" block
                            style={{ backgroundColor: '#121212' }}>
                            登录
                        </Button>
                    </Form.Item>

                    <Form.Item style={{ marginBottom: 0 }}>
                        <Text type="secondary">
                            <Button
                                type="link"
                                size="small"
                                onClick={handleForgotPassword}
                                style={{ float: "right", color: "#121212" }}
                            >
                                忘记密码？
                            </Button>
                        </Text>
                    </Form.Item>
                </Form>
            </div>
        )
    }
    return (
        <div>
            <Form
                form={form}
                name="findPassword"
                onFinish={handlefindPassword}
                layout="vertical"
                autoComplete="off"
            >
                <Form.Item
                    name="email"
                    label="邮箱"
                    rules={[{ required: true, type: "email", message: "请输入有效邮箱" }]}
                >
                    <Input placeholder="请输入邮箱" />
                </Form.Item>
                <Form.Item
                    name="password"
                    label="新密码"
                    rules={[
                        { required: true, message: "请输入新密码" },
                        { min: 6, message: "密码至少6个字符!" },
                    ]}
                >
                    <Input.Password placeholder="请输入新密码" />
                </Form.Item>
                <Form.Item
                    name="confirmPassword"
                    label="确认密码"
                    rules={[
                        { required: true, message: "请输入确认密码" },
                        { min: 6, message: "密码至少6个字符!" },
                    ]}
                >
                    <Input.Password placeholder="请输入确认密码" />
                </Form.Item>
                <Form.Item
                    label="验证码"
                    required
                    rules={[{ required: true, message: "请输入验证码" }]}
                >
                    <Space.Compact>
                        <Form.Item
                            name="code"
                            noStyle
                            rules={[{ required: true, message: "请输入验证码" }]}
                        >
                            <Input.OTP
                                formatter={(str) => str.toUpperCase()}
                                style={{ flex: 1 }}
                            />
                        </Form.Item>
                        <Button
                            type="link"
                            onClick={handleGetCode}
                            disabled={countdown > 0}
                            style={{ color: "#121212" }}
                        >
                            {countdown > 0 ? `(${countdown})秒后重新发送` : '发送验证码'}
                        </Button>
                    </Space.Compact>
                </Form.Item>

                <Form.Item>
                    <Space>
                        {/* 这里发送完可以显示一个多少秒后继续发送 */}
                        <Button type="primary" htmlType="submit" block
                            style={{ backgroundColor: '#121212' }}>
                            确认修改密码
                        </Button>
                        <Button htmlType="button">
                            {/* onClick={onReset} */}
                            重置
                        </Button>
                    </Space>
                </Form.Item>
                <Form.Item style={{ marginBottom: 0 }}>
                    <Text type="secondary">
                        <Button
                            type="link"
                            size="small"
                            onClick={handleBackToLogin}
                            style={{ float: "right", color: "#121212" }}
                        >
                            返回登录
                        </Button>
                    </Text>
                </Form.Item>
            </Form>
        </div>
    );
}



export default Loginform;