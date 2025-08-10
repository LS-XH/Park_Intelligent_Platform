package com.dyy.service.impl;

import com.dyy.domain.Result;
import com.dyy.domain.Users;

import java.util.Map;

public interface UsersService {
    Map<String,Object> loginByPassword(String email, String password);

    Result register(Users user, String code);

    Result sendCodeByEmail(String email);

    Result getUser(Long id);

    Result findPassword(Users user, String code);
}
