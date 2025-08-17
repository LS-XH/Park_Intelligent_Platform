import axios from "axios";

const apiClient = axios.create({
    baseURL: 'http://47.113.224.195:30428',
    timeout: 10000,
    headers: {
        "Content-Type": "application/json",
    },
});


export default apiClient;