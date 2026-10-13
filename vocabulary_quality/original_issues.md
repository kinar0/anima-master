# 原有词库初筛问题清单

完整逐行审查记录见 original_audit.json。原值全部保留。问题标记分为确定不等价、修饰条件丢失、语境风险和目录缺失；语境风险不等于该词条在所有使用中错误。

用户本轮确认 `事后=aftersex` 在本词库语境中正确，已认可保留，不列为问题。

| 行 | 原映射 | 类型 | 依据 | 邻近反例 |
| --- | --- | --- | --- | --- |
| 1 | `短裙=skirt` | lost_modifier | skirt 没编码短的条件 | 短裙与长裙不同 |
| 29 | `丝袜=stocking` | catalog_missing | stocking 未在本地目录精确命中；袜型也不明确 | 连裤与独立长袜需区别 |
| 38 | `中性=androgynous` | context_sensitive | 性别外观语境成立，配色或语气语境不成立 | 中性灰配色 |
| 43 | `丰满=plump` | context_sensitive | 可指胸部或体型，plump 固定了体型解释 | 胸部丰满但腰身纤细 |
| 84 | `低头=looking_down` | context_sensitive | 头姿不保证眼睛向下看 | 低头但抬眼看观众 |
| 89 | `侧头=head_tilt` | context_sensitive | 倾斜与转头侧向有别 | 转头看侧面 |
| 95 | `侧身=on_side` | context_sensitive | on_side 是侧卧姿态，不是站着侧身 | 侧身站立 |
| 130 | `冠冕=tiara` | context_sensitive | tiara 结构不覆盖所有冠冕 | 封闭王冠 |
| 179 | `卷发=wavy_hair` | context_sensitive | 卷曲强度多样，wavy_hair 固定波浪发 | 紧密小卷 |
| 188 | `双手相握=own_hands_together` | context_sensitive | own_hands_together 是自己的双手，不是两个人握手 | 两人双手相握 |
| 198 | `双辫=twin_braids` | context_sensitive | twin_braids 不一定覆盖两束非编织发束 | 两束直发 |
| 217 | `变装=crossdressing` | context_sensitive | 换装泛词不一定异性装扮 | 从校服变装为同性交际礼服 |
| 222 | `另一人=1other` | context_sensitive | 1other 是性别计数，不是泛指又一位人物 | 另一人是女孩 |
| 229 | `吊带=suspenders` | context_sensitive | 可指背心肩带，不一定背带 suspenders | 细吊带连衣裙 |
| 236 | `吐舌头=:p` | added_condition | :p 带表情符号/嘴形语境，吐舌本身不保证该形状 | 张大嘴伸出舌头 |
| 239 | `呼吸=breath` | context_sensitive | 一般呼吸不必有可见 breath | 静止肖像里平稳呼吸 |
| 269 | `外套=jacket` | context_sensitive | 泛称包含大衣、披风等，不总是 jacket | 长款大衣外套 |
| 313 | `套装=suit` | context_sensitive | 不总是西式 suit | 运动套装 |
| 341 | `小孩=child` | context_sensitive | 词义成立但名字、称呼或比喻不能变成年龄事实 | 成年角色被昵称为小孩 |
| 366 | `开口=open_mouth` | context_sensitive | 衣物开口或发言不保证张嘴状态 | 胸前开口连衣裙 |
| 430 | `手腕袖口=wrist_cuffs` | unnatural | 中文表达生硬，袖口与独立 wrist_cuffs 要区别 | 普通衬衫袖口 |
| 491 | `敞开外套=open_coat` | context_sensitive | 外套泛词固定 open_coat | 敞开的夹克 |
| 492 | `敞开的外套=open_jacket` | context_sensitive | 外套泛词固定 open_jacket，与上一条近同义却不同 | 敞开的大衣 |
| 523 | `星星=star_(symbol)` | context_sensitive | star_(symbol) 是符号，不必是天空恒星 | 夜空中的星星 |
| 621 | `浴袍=robe` | context_sensitive | robe 泛袍不保证浴袍条件 | 浴后浴袍 |
| 675 | `玩偶=stuffed_animal` | context_sensitive | 不总是毛绒动物，可为人形 doll | 硬质人形玩偶 |
| 694 | `白丝袜=white_pantyhose` | context_sensitive | 白色丝袜不一定连裤袜 | 白色独立长袜 |
| 700 | `白皮肤=white_skin` | context_sensitive | white_skin 可能比浅色肤色更特异，需要核对 | 普通浅肤色 |
| 724 | `白裙子=white_skirt` | context_sensitive | 口语裙子可指连衣裙，white_skirt 固定半身裙 | 白色连衣裙 |
| 740 | `眨眼=wink` | context_sensitive | 一般双眼眨动不是单眼 wink | 两只眼同时眨一下 |
| 871 | `耳钉=ear_piercing` | lost_object | ear_piercing 编码穿孔，不编码耳钉形制 | 耳孔与耳钉区别 |
| 890 | `背心=tank_top` | context_sensitive | 可指 vest 或 sweater_vest，不能都当 tank_top | 西装针织背心 |
| 899 | `胸部=breasts` | context_sensitive | 不必是乳房 breasts，可为男角色胸口 | 胸部护甲 |
| 999 | `裙子=skirt` | context_sensitive | 裙子口语形制不确定 | 连衣裙也常叫裙子 |
| 1001 | `裙底=upskirt` | context_sensitive | 物件部位不必是向裙底拍摄 | 裙底缝着蕾丝 |
| 1022 | `说出心声=spoken_heart` | added_condition | spoken_heart 需要心形符号，不是泛指真心话 | 说出自己的担忧 |
| 1026 | `走光=pantyshot` | added_condition | 暴露部位未指定，pantyshot 固定内裤 | 肩带滑落走光 |
| 1030 | `趴=on_stomach` | short_alias | 单字子串可命中字面提及和其他词 | 趴这个字写在纸上 |
| 1050 | `过时=dated` | context_sensitive | 画面年代感 dated 不等于物件过时 | 过时的技术 |
| 1091 | `长手套=elbow_gloves` | added_condition | 长不保证 elbow_gloves 到肘长度 | 过腕但不到肘的手套 |
| 1100 | `闪烁=sparkle` | context_sensitive | 灯光闪动不必有 sparkle 星芒 | 屏幕忽明忽暗 |
| 1116 | `露出=exposure` | short_alias | 任何露出都会触发 exposure；本地目录也缺失 | 露出微笑 |
| 1136 | `项圈=choker` | context_sensitive | choker 常为贴颈饰品，动物项圈另有 collar | 狗项圈 |
| 1157 | `马赛克=censored` | context_sensitive | 可为镶嵌图案，不一定 censored | 马赛克地砖 |
| 1175 | `鸭舌帽=mob_cap` | non_equivalent | mob_cap 是褶边软帽，不是鸭舌帽 | 普通鸭舌帽 |
| 1184 | `黑丝袜=black_stocking` | catalog_missing | black_stocking 无本地精确命中；袜型待明确 | 黑色连裤袜与大腿袜 |
| 1207 | `黑色紧身连衣裤=black_leotard` | context_sensitive | leotard 不覆盖带长裤腿的连体服 | 全长连体裤 |
| 1263 | `运动服=gym_uniform` | context_sensitive | 泛运动服不都属于 gym_uniform | 职业队训练服 |
| 1275 | `高帽=top_hat` | context_sensitive | 形容帽子高不保证 top_hat 礼帽 | 高尖女巫帽 |
| 1291 | `认真=serious` | context_sensitive | 行为态度不等于严肃表情 | 认真看比赛但笑着 |
| 1302 | `紫色上衣=purple_shirt` | context_sensitive | 上衣泛指不一定 shirt | 紫色毛衣 |
| 1336 | `三人行=threesome` | context_sensitive | 既可性语境，也可三人同行，不能无条件判定 | 三人一起散步 |
| 1382 | `白色紧身连衣裤=white_leotard` | context_sensitive | leotard 不覆盖带长裤腿的连体服 | 全长连体裤 |
| 1399 | `骷髅=skull` | context_sensitive | 可能指完整骨架，不只是 skull | 完整人形骷髅 |
