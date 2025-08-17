import CryptoJS from 'crypto-js';
import forge from 'node-forge';

// AES 加密函数
export const encryptWithAES = (data, keyBase64) => {
    try {
        // 将Base64密钥转为CryptoJS格式
        const key = CryptoJS.enc.Base64.parse(keyBase64);

        // 生成随机IV（16字节）
        const iv = CryptoJS.lib.WordArray.random(16);

        // 加密配置（明确指定参数）
        const encrypted = CryptoJS.AES.encrypt(data, key, {
            iv: iv,
            mode: CryptoJS.mode.CBC,
            padding: CryptoJS.pad.Pkcs7
        });

        // 组合IV和密文，输出Base64
        return CryptoJS.enc.Base64.stringify(iv.concat(encrypted.ciphertext));
    } catch (error) {
        console.error('AES 加密失败:', error);
        throw error;
    }
};

// AES 解密函数
export const decryptWithAES = (ciphertext, key) => {
    try {
        // 1. Base64解码获取完整数据
        const encryptedBytes = CryptoJS.enc.Base64.parse(ciphertext);

        // 2. 提取IV（前16字节）
        const iv = CryptoJS.lib.WordArray.create(
            encryptedBytes.words.slice(0, 4), // 16 bytes = 4 words
            16
        );

        // 3. 提取密文（剩余部分）
        const ciphertextBytes = CryptoJS.lib.WordArray.create(
            encryptedBytes.words.slice(4),
            encryptedBytes.sigBytes - 16
        );

        // 4. 将字符串密钥转为CryptoJS格式
        const keyBytes = CryptoJS.enc.Base64.parse(key);

        // 5. 解密
        const decrypted = CryptoJS.AES.decrypt(
            { ciphertext: ciphertextBytes },
            keyBytes,
            {
                iv: iv,
                mode: CryptoJS.mode.CBC,
                padding: CryptoJS.pad.Pkcs7
            }
        );
        console.log("Decrypted raw bytes:", decrypted.toString(CryptoJS.enc.Hex))
        return decrypted.toString(CryptoJS.enc.Utf8);
    } catch (error) {
        console.error('AES 解密失败:', error);
        throw error;
    }
};

// 生成随机 AES 密钥
export const generateAESKey = () => {
    // 生成指定长度的随机字节数组
    const wordArray = CryptoJS.lib.WordArray.random(32);
    // 转换为 base64 格式字符串
    return CryptoJS.enc.Base64.stringify(wordArray);
};
// RSA 加密函数
export const encryptWithRSA = (data, publicKeyPem) => {
    try {
        const publicKey = forge.pki.publicKeyFromPem(publicKeyPem);

        // 显式指定 OAEP 参数：SHA-256
        const encrypted = publicKey.encrypt(data, 'RSA-OAEP', {
            md: forge.md.sha256.create(),
            mgf1: {
                md: forge.md.sha256.create()
            }
        });

        return forge.util.encode64(encrypted);
    } catch (error) {
        console.error('RSA 加密失败:', error);
        throw error;
    }
};

// RSA 解密函数 用一开始给的私钥来解密密码
export const decryptWithRSA = (encryptedData, privateKeyPem) => {
    try {
        console.log("", privateKeyPem);
        console.log("", encryptedData);


        const privateKey = forge.pki.privateKeyFromPem(privateKeyPem);
        const encryptedBytes = forge.util.decode64(encryptedData);

        // 显式指定与后端一致的 OAEP 参数
        const decrypted = privateKey.decrypt(encryptedBytes, 'RSA-OAEP', {
            md: forge.md.sha256.create(), // SHA-256 哈希
            mgf1: {
                md: forge.md.sha256.create() // MGF1 使用 SHA-256
            }
        });
        console.log("decrypted:", decrypted);

        return decrypted;
    } catch (error) {
        console.error('RSA 解密失败:', error);
        throw error;
    }
};

// 结合 AES 和 RSA 的加密函数
// 1. 生成随机 AES 密钥
// 2. 使用 AES 密钥加密数据
// 3. 使用 RSA 公钥加密 AES 密钥
// 4. 返回加密后的数据和加密后的密钥
export const encryptWithAESAndRSA = (
    data,
    rsaPublicKey
) => {
    try {
        // 生成随机 AES 密钥
        const aesKey = generateAESKey();

        console.log("aesKey:", aesKey);

        // 使用 AES 密钥加密数据
        const encryptedData = encryptWithAES(data, aesKey);

        // 使用 RSA 公钥加密 AES 密钥
        const encryptedKey = encryptWithRSA(aesKey, rsaPublicKey);

        return {
            encryptedData,
            encryptedKey
        };
    } catch (error) {
        console.error('AES+RSA 加密失败:', error);
        throw error;
    }
};

// 结合 AES 和 RSA 的解密函数
// 1. 使用 RSA 私钥解密 AES 密钥
// 2. 使用解密后的 AES 密钥解密数据
export const decryptWithAESAndRSA = (
    encryptedData,
    encryptedKey,
    rsaPrivateKey
) => {
    try {
        // 使用 RSA 私钥解密 AES 密钥
        const aesKey = decryptWithRSA(encryptedKey, rsaPrivateKey);

        // 使用解密后的 AES 密钥解密数据
        const decryptedData = decryptWithAES(encryptedData, aesKey);

        return decryptedData;
    } catch (error) {
        console.error('AES+RSA 解密失败:', error);
        throw error;
    }
};