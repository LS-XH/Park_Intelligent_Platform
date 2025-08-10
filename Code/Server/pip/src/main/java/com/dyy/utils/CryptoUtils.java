package com.dyy.utils;

import com.dyy.dto.EncryptionResultDTO;
import com.dyy.dto.EncryptionResultDTO;
import lombok.extern.slf4j.Slf4j;

import java.util.Base64;
@Slf4j
public class CryptoUtils {

    /**
     * 双重加密（与前端 encryptWithAESAndRSA 对应）
     * @param data 原始数据
     * @param rsaPublicKey RSA 公钥（PEM 格式）
     * @return 包含加密数据和加密密钥的对象
     */
    public static EncryptionResultDTO encryptWithAESAndRSA(String data, String rsaPublicKey) throws Exception {
        // 生成随机 AES 密钥（32字节）
        String aesKey = generateRandomAESKey(32);

        // 用 AES 加密数据
        String encryptedData = AesUtils.encrypt(data, aesKey);

        // 用 RSA 加密 AES 密钥
        String encryptedKey = RsaUtils.encrypt(aesKey, rsaPublicKey);

        EncryptionResultDTO encryptionResult = new EncryptionResultDTO(encryptedData, encryptedKey);
        return encryptionResult;
    }

    /**
     * 双重解密（与前端 decryptWithAESAndRSA 对应）
     */
    public static String decryptWithAESAndRSA(String encryptedData, String encryptedKey, String rsaPrivateKey) throws Exception {
        // 用 RSA 解密 AES 密钥
        String aesKey = RsaUtils.decrypt(encryptedKey, rsaPrivateKey);
        log.info("解密后的AES密钥：" + aesKey);
        // 用 AES 解密数据
        return AesUtils.decrypt(encryptedData, aesKey);
    }

    public static String generateRandomAESKey(int keyLength) {
        if (keyLength != 16 && keyLength != 24 && keyLength != 32) {
            throw new IllegalArgumentException("Invalid key length. Must be 16, 24, or 32 bytes");
        }
        byte[] key = new byte[keyLength];
        new java.security.SecureRandom().nextBytes(key);
        return Base64.getEncoder().encodeToString(key); // 完整Base64编码
    }

}
