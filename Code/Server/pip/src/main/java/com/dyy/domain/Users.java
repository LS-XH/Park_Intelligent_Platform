package com.dyy.domain;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Users {
    @TableId(value = "id",type = IdType.AUTO)
    Long id;
    String username;
    String password;
    String email;
    String createdTime;
}
