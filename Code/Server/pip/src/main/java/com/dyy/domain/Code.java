package com.dyy.domain;

public class Code {
    /**
     * 2**
     * 响应成功
     */
    public static final Integer SUCCESS = 200;      // 响应成功
    public static final Integer CREATED = 201;      // 资源创建成功

    /**
     * 4**
     * 客户端错误
     */
    public static final Integer BAD_REQUEST = 400;  // 请求参数错误
    public static final Integer UNAUTHORIZED = 401; // 未认证
    public static final Integer FORBIDDEN = 403;    // 无权限访问
    public static final Integer NOT_FOUND = 404;    // 找不到该资源
    public static final Integer CONFLICT = 409;     // 资源冲突

    /**
     * 5**
     * 服务器错误
     */

    public static final Integer INTERNAL_ERROR = 500;      // 服务器内部错误
    public static final Integer FEATURES_ARE_NOT_DEVELOPED = 501;      // 功能未开发
    public static final Integer BAD_GATEWAY = 502;         // 网关错误
}
