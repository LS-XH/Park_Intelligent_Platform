package com.dyy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.dyy.domain.Users;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface UsersMapper extends BaseMapper<Users> {
}
