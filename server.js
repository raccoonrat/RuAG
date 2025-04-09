const express = require('express');
const bodyParser = require('body-parser');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

// 中间件
app.use(cors());
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'demo')));

// 模拟数据
const mockRules = [
    {
        name: "哈兰德规则",
        condition: "entity_a_哈兰德",
        prediction: "relation_效力于",
        accuracy: 1.0,
        translated: "当实体A为哈兰德时，存在效力于关系（准确度：1.00）。"
    },
    {
        name: "姆巴佩规则",
        condition: "entity_a_姆巴佩",
        prediction: "relation_效力于",
        accuracy: 1.0,
        translated: "当实体为姆巴佩时，可以确定其效力于关系，精确度为1.00。"
    },
    {
        name: "梅西规则",
        condition: "entity_a_梅西",
        prediction: "relation_效力于",
        accuracy: 1.0,
        translated: "当检测到实体A为"梅西"时，可以确定其存在"效力于"的关联关系，该判断的置信度为100%。"
    }
];

// API路由
app.post('/api/run', (req, res) => {
    const { task, model, query } = req.body;
    
    // 模拟处理延迟
    setTimeout(() => {
        // 返回模拟数据
        res.json({
            task,
            model,
            query,
            rules: mockRules,
            original_response: "根据文本分析，梅西是一名足球运动员，曾效力于巴塞罗那足球俱乐部和巴黎圣日耳曼足球俱乐部，目前效力于迈阿密国际足球俱乐部。",
            enhanced_response: "根据文本分析，我识别到以下实体关系：\n\n实体：梅西（人物）\n实体：巴塞罗那足球俱乐部（组织）\n关系：效力于（历史）\n\n实体：梅西（人物）\n实体：巴黎圣日耳曼足球俱乐部（组织）\n关系：效力于（历史）\n\n实体：梅西（人物）\n实体：迈阿密国际足球俱乐部（组织）\n关系：效力于（当前）\n\n以上关系符合规则：当检测到实体A为"梅西"时，可以确定其存在"效力于"的关联关系。",
            is_valid: true,
            violations: []
        });
    }, 1500);
});

// 启动服务器
app.listen(PORT, () => {
    console.log(`RuAG PoC 服务器运行在 http://localhost:${PORT}`);
});