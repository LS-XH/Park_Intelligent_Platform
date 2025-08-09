import apiClient from "../index.jsx";

//用户获取验证码
export const getCodeAPI = async (email) => {
    const response = await apiClient.post("/user/getCode", { email });
    return response;
}
