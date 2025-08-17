import apiClient from "../index.jsx";


//登录路由
export const login = async (encryptedData, encryptedKey) => {
    const body = {
        encryptedData,
        encryptedKey
    };
    const response = await apiClient.post("/users/password", body);
    return response;
};


//找回密码
export const findPassword = async (encryptedData, encryptedKey) => {
    const body = {
        encryptedData,
        encryptedKey
    }
    const response = await apiClient.put(`/users/findPassword`, body);
    return response;
};

//注册路由
export const registerAPI = async (
    encryptedData, encryptedKey
) => {
    const body = {
        encryptedData,
        encryptedKey
    }
    const response = await apiClient.post("/users/register", body);
    return response;
};

//获取验证码
export const getCodeAPI = async (encryptedData, encryptedKey) => {
    const response = await apiClient.get("/users/sendCodeByEmail", {
        params: {
            encryptedData,
            encryptedKey
        }
    });
    return response;
};