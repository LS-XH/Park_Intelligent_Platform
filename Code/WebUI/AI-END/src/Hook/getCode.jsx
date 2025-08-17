import { useState, useEffect } from "react";
import { message } from "antd";
// import '@ant-design/v5-patch-for-react-19';
import { getCodeAPI } from "../api/service/registerService"; // Adjust the import path as necessary
import { encryptWithAESAndRSA } from "../../util/entrypt"; // Adjust the import path as necessary

//=====复用获取验证码逻辑的hook=====

//加密公钥
const publicKey = `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAsQBfHrU7NYVB8l0kmD79ayRbS2Nmu0gOKIg177flG/MiZd5TIYuH+eOINrFFgu6K1jmTqeUDw5Lm2SPofC1fV++V6yhJu8Vveaa0WhFElSrp5F4vsZ34HB7kpZmH6Vp/u9tdohDrXe+cVdO74ILxsw9CLpEpFrFHmgThVSKtNfwCExZeOT5lN6UKgsxp+HIFbhKWF9NMpmeYw5ie10YevN9Fq9x11aeg+ZgKct1GzF9RfOcX0h6Mz4xu45q5bWRQS+djvprBS5tvYOCVZj9KEanltbFFq71PmiQLdkH7imCFtwHPZzK5TAYeknH+raSjlGDMsijs+I8tR8XpuQcXtwIDAQAB
-----END PUBLIC KEY-----`;


//解密私钥
const privateKey = `-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC1AZj7Ik201dBfzZ9eP3rJCsdt9Hy7RP9KR+xTWnI+VGrW+fDUAzYyFzegXemaXpgePTchdbeioow+JkUPkWQWOmOlmgg+aXMymqgvkix1e3MrepRJkRNVXJJ5KvSJt7OPR5K5B8QTyLXEP8sOLaVpfTJxOmrbe5EwLf1iF63JmL+JoM3vxFZX/kroirb9fYCsWfaZG8IIil5SXlgS7UPd5mpy+pXn0QrhwP2Gw23nG8pX+AwTuL65dJC8DSZMfvFdWZeTQnw6AVS8StMmvXazMNWgOzA65kCJlQB62T69/CYttAjGGXzbWZ6Koe0/TOtpADIdLDOPIe++lxm6Rrn9AgMBAAECggEASPwDbOPIlG2Qb0jQhWawQkc74cyuzK4GCDQXCQcTwKk2SUePwVUoMatl7R5g9rNEwBCr3ayDJqtHRDoXJ69WvZW+n0QMJepMHm/49/GHRrnH1xS+nSlHs+g3UW8uGie92byg30XP3LBWBnM4k5d5Np9aSwiklKpvAQ/SNw7YLsxgG3tjmMgzbzQnNtTdW81BV+A4KIaYVUZnoSEVSzHN3T7WZgG9TLiCm7AowQ8qFTK0Sxu0kO3JGc6G3GkQmR77J4YDIv8O+Da+ITmyVEwxtzuIKNa/VDCtV3Anxit+Hk0xBNsT9Vvdv3upMyjOggjXnWbQYUXN4zbv1IoWlwqOiwKBgQDlzIUZHQPY5t1BWOOXr1w7KRVfxtqm9fKy1x4v+T7f1SPOmFpmAg3Qfv6c1dpON8d6NJS01kuymFW/iPVdvjpoGWqYBSEn0mbz9iBU9quNaU4WhqmlU8OdfqCch0jOK/l+3WCl4BshShpis2mvXPa8HtygkRqQfMVGML2nGttb+wKBgQDJpOTiJSRlcRgj27qXsQjXwrJsEAoZ0cA/UaUlWed/qHBsLE9yk+Y3bxlFc8Sf0wLJejaaCOQ4IS6e81bl7AOT/VBqU77zzBS1uZ4L1dMlJipELgtQcsv06CblDhnRJLAAJ0/xtX6HUqu2v8pGqaLyqsESZXn8TdwxyBKfbnioZwKBgQCIgQ7nNhcc9zajJLw9VIvDEMqDlEo6N4sttR9XfAVfTOryRAoe4kV2fpmcbGQ7ZmL2MtnK+ikJM/hryF2IjAGB6Ocq2pExaIiDjsbx8X1CiTU7qE6JyNJAcgHSOYKEBhc0xygsII29HpnB27WB2AUxBlwkfU18WsGMylM+OnPnlQKBgQC7MtQyhlzVuDq6/4Co1vfopp3R6MoX0jxyDDAPDvn177//DNvs+RVfHUsOyT0fS1xpA4axVdPZsCSB+FMSPRvNRfxj2b+Kwknvs5TgU/AjqtzOUxi55PkoMmX5fC/HlBG48sYrFV2T79HuZPs6wr2+H3wCwiaPbxEfPijbzklBvQKBgBFCAippRBOSX6gol9VJSwzB61Ak9U2vYKucWia4GrEtS/faEUmNKj220qksTEjiACnbTrWojZFKEWx30s+mrkwXdXmNbuq2so5fEvjGQ8rKXCcJNp5/pInPMvhCvw4tUuJ9lEH8EXkDFFRlTVoUnlWofHdTcQreOs/tkHlfrC5M
-----END PRIVATE KEY-----`
export const useCode = () => {
    const [countdown, setCountdown] = useState(0);

    //点击发送验证码获取验证码
    const sendCode = async (email) => {
        if (!email) {
            message.error("请输入邮箱");
            return;
        }

        const { encryptedData, encryptedKey } = encryptWithAESAndRSA(
            JSON.stringify({ email: email }),
            publicKey
        );
        console.log("找回密码");


        const response = await getCodeAPI(encryptedData, encryptedKey);

        if (response.code === 200) {
            setCountdown(60);
            message.success("验证码已发送");
        } else {
            message.error("验证码发送失败");
        }
    };

    //点击发送验证码的倒计时
    useEffect(() => {
        if (countdown > 0) {
            const timer = setTimeout(() => setCountdown(c => c - 1), 1000);
            return () => clearTimeout(timer);
        }
    }, [countdown]);

    //返回倒计时和发送验证码的函数
    return {
        countdown,
        sendCode,
    };
};