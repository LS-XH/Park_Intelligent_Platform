package com.dyy.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class CorsConfig implements WebMvcConfigurer {
    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/**")  // 允许所有接口
                .allowedOrigins("*")  // 允许所有前端域名（测试用，生产环境需指定具体域名）
                .allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS")  // 允许的请求方法
                .allowedHeaders("*")  // 允许的请求头
                .exposedHeaders("Authorization", "Rsakey") // 明确暴露自定义头
                .maxAge(3600);  // 预检请求缓存时间（秒）
    }
}
