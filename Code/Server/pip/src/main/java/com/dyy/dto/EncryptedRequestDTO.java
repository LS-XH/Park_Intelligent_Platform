package com.dyy.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class EncryptedRequestDTO {
    private String encryptedData; // AES加密的业务数据
    private String encryptedKey;  // RSA加密的AES密钥
}

