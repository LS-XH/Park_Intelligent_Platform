package com.dyy.utils;

import lombok.extern.slf4j.Slf4j;

import javax.crypto.Cipher;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;
import java.util.Arrays;
import java.util.Base64;
@Slf4j
public class AesUtils {

    private static final String ALGORITHM = "AES";
    private static final String TRANSFORMATION = "AES/CBC/PKCS5Padding";

    /**
     * AES 加密（与前端 CryptoJS 默认模式兼容）
     * @param data 明文数据
     * @param keyBase64 密钥（需与前端一致）
     * @return Base64 编码的加密结果
     */
    public static String encrypt(String data, String keyBase64) throws Exception {
        try {
            log.info("加密密钥： {}", keyBase64);
            // 1. Base64解码密钥（关键修改！）
            byte[] keyBytes = Base64.getDecoder().decode(keyBase64);
            if (keyBytes.length != 16 && keyBytes.length != 24 && keyBytes.length != 32) {
                throw new IllegalArgumentException("Invalid AES key length: " + keyBytes.length);
            }

            // 2. 生成随机IV
            byte[] ivBytes = new byte[16];
            new SecureRandom().nextBytes(ivBytes);

            // 3. 使用解码后的密钥字节
            SecretKeySpec secretKey = new SecretKeySpec(keyBytes, ALGORITHM);
            Cipher cipher = Cipher.getInstance(TRANSFORMATION);
            cipher.init(Cipher.ENCRYPT_MODE, secretKey, new IvParameterSpec(ivBytes));

            // 4. 加密数据
            byte[] encrypted = cipher.doFinal(data.getBytes(StandardCharsets.UTF_8));

            // 5. 组合IV+密文
            byte[] combined = new byte[ivBytes.length + encrypted.length];
            System.arraycopy(ivBytes, 0, combined, 0, ivBytes.length);
            System.arraycopy(encrypted, 0, combined, ivBytes.length, encrypted.length);

            return Base64.getEncoder().encodeToString(combined);
        } catch (Exception e) {
            throw new RuntimeException("AES 加密失败", e);
        }
    }

    /**
     * AES 解密（兼容前端 CryptoJS 加密结果）
     * @param encryptedData Base64 编码的加密数据
     * @param keyBase64 密钥
     * @return 解密后的明文
     */

    // 解密方法（匹配前端 CryptoJS 的默认加密方式）
    public static String decrypt(String encryptedData, String keyBase64) throws Exception {
        try {

            // 1. Base64解码密钥
            byte[] keyBytes = Base64.getDecoder().decode(keyBase64);
            if (keyBytes.length != 32) {
                throw new IllegalArgumentException("AES-256需要32字节密钥");
            }

            // 2. Base64解码加密数据
            byte[] encryptedBytes = Base64.getDecoder().decode(encryptedData);
            if (encryptedBytes.length < 16) {
                throw new IllegalArgumentException("数据太短，缺少IV");
            }

            // 3. 提取IV（前16字节）
            byte[] iv = Arrays.copyOfRange(encryptedBytes, 0, 16);

            // 4. 提取密文（剩余部分）
            byte[] ciphertext = Arrays.copyOfRange(encryptedBytes, 16, encryptedBytes.length);

            // 5. 初始化解密器
            Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            cipher.init(
                    Cipher.DECRYPT_MODE,
                    new SecretKeySpec(keyBytes, "AES"),
                    new IvParameterSpec(iv)
            );

            // 6. 执行解密
            byte[] decrypted = cipher.doFinal(ciphertext);
            return new String(decrypted, StandardCharsets.UTF_8);
        } catch (Exception e) {
            throw new RuntimeException("AES解密失败: " + e.getMessage(), e);
        }
    }
}
