package com.dyy.utils;

import cn.hutool.crypto.digest.BCrypt;

public class HashSaltUtil {
    /**
     * @param plainPassword
     * @return 哈希密码
     */
    public static String creatHashPassword(String plainPassword) {
        return BCrypt.hashpw(plainPassword, BCrypt.gensalt());
    }

    /**
     * @param plainPassword
     * @param hashedPassword
     * @return boolean
     * 比较明文密码与哈希密码
     */
    public static boolean verifyHashPassword(String plainPassword, String hashedPassword) {
        return BCrypt.checkpw(plainPassword, hashedPassword);
    }


    /**
     * @param password
     * @return boolean
     * 判断是否为明文密码
     */
    public static boolean isPlainPassword(String password) {
        return password.length() < 60;
    }
}
