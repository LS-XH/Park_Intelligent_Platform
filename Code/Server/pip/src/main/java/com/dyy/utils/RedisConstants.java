package com.dyy.utils;

public class RedisConstants {
    /**
     * 验证码
     */
    public static final String LOGIN_CODE_KEY = "login:code:";
    public static final Long LOGIN_CODE_TTL = 5L;

    /**
     * token验证
     * tokenkey的头
     * token有效期
     */
    public static final String LOGIN_USER_KEY = "login:user:";
    public static final long LOGIN_USER_TTL = 30L;

    /**
     * 错误 重复
     */
    public static final String ERROR_REPEAT_KEY = "error:repeat:";
    public static final Integer MAX_ERROR_TIME = 10;

    /**
     * 通知是否已读
     */
    public static final Integer IS_READ = 1;
    public static final Integer IS_NOT_READ = 0;

    /**
     * 通知发送者是否存在
     */
    public static final Integer IS_SENDER_EXIST = 1;
    public static final Integer IS_SENDER_NOT_EXIST = 0;
}
