package com.dyy.utils;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.mail.MailSendException;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Service;

@Service
public class EmailService {

    @Autowired
    private JavaMailSender mailSender;

    @Value("${spring.mail.username}")
    private String MY_EMAIL;

    public boolean sendSimpleEmail(String to, String subject, String content) {
        SimpleMailMessage message = new SimpleMailMessage();
        message.setFrom(MY_EMAIL); // 发件人邮箱
        message.setTo(to);                   // 收件人邮箱
        message.setSubject(subject);         // 邮件主题
        message.setText(content);            // 邮件内容

        try {
            mailSender.send(message);
            System.out.println("邮件发送成功，收件人：" + to);
            return true; // 发送成功
        } catch (MailSendException e) {
            // 处理邮件发送失败的异常（重点捕获550错误）
            System.err.println("邮件发送失败，收件人：" + to + "，错误信息：" + e.getMessage());

            // 检查异常中是否包含"550"（收件人不存在的典型错误码）
            if (e.getMessage() != null && e.getMessage().contains("550")) {
                System.err.println("原因：收件人邮箱可能不存在或无效");
            }
            return false; // 发送失败（包括邮箱不存在的情况）
        } catch (Exception e) {
            // 处理其他可能的异常（如配置错误、网络问题等）
            System.err.println("邮件发送发生未知异常，收件人：" + to + "，错误：" + e.getMessage());
            return false;
        }
    }
}
