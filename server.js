const express = require('express');
const bodyParser = require('body-parser');
const path = require('path');
const cors = require('cors');
require('dotenv').config();

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
        translated: "当检测到实体A为\"梅西\"时，可以确定其存在\"效力于\"的关联关系，该判断的置信度为100%。"
    }
];

// API路由
function getLLMProvider(modelName) {
    if (modelName === "mock") {
        return { 
            generate: (query, context) => "这是模拟LLM生成的响应",
            name: "Mock LLM",
            apiKey: "不需要API_KEY"
        };
    } else if (modelName === "gpt-3.5-turbo" || modelName === "gpt-4") {
        return {
            generate: async (query, context) => {
                // 这里实现OpenAI API调用
                return "这是GPT生成的响应";
            },
            name: modelName,
            apiKey: process.env.OPENAI_API_KEY || "未设置OPENAI_API_KEY"
        };
    } else if (modelName === "deepseek") {
        return {
            generate: async (query, context) => {
                // 这里实现DeepSeek API调用
                return "这是DeepSeek生成的响应";
            },
            name: "DeepSeek",
            apiKey: process.env.DEEPSEEK_API_KEY || "未设置DEEPSEEK_API_KEY"
        };
    } else if (modelName === "volc-ark-deepseek") {
        return {
            generate: async (query, context) => {
                // 这里实现火山方舟DeepSeek API调用
                return "这是火山方舟DeepSeek生成的响应";
            },
            name: "火山方舟DeepSeek",
            apiKey: process.env.ARK_API_KEY || "未设置ARK_API_KEY"
        };
    } else {
        return {
            generate: (query, context) => "未知模型，使用模拟响应",
            name: "Unknown Model",
            apiKey: "未知模型"
        };
    }
}

// 在文件顶部添加监控相关模块
const prometheus = require('prom-client');
const responseTime = require('response-time');

// 初始化监控指标
const httpRequestDurationMicroseconds = new prometheus.Histogram({
    name: 'http_request_duration_ms',
    help: 'Duration of HTTP requests in ms',
    labelNames: ['method', 'route', 'code'],
    buckets: [0.1, 5, 15, 50, 100, 200, 300, 400, 500]
});

const activeRequests = new prometheus.Gauge({
    name: 'node_active_requests',
    help: 'Number of active requests'
});

const totalRequests = new prometheus.Counter({
    name: 'node_total_requests',
    help: 'Total number of requests'
});

const errorRequests = new prometheus.Counter({
    name: 'node_error_requests',
    help: 'Total number of error requests'
});

// 在Express应用初始化后添加中间件
app.use(responseTime((req, res, time) => {
    httpRequestDurationMicroseconds
        .labels(req.method, req.route.path, res.statusCode)
        .observe(time);
}));

app.use((req, res, next) => {
    activeRequests.inc();
    totalRequests.inc();
    next();
});

app.use((req, res, next) => {
    res.on('finish', () => {
        activeRequests.dec();
        if (res.statusCode >= 400) {
            errorRequests.inc();
        }
    });
    next();
});

// 添加监控端点
app.get('/metrics', async (req, res) => {
    res.set('Content-Type', prometheus.register.contentType);
    res.end(await prometheus.register.metrics());
});

// 在API路由中添加自定义指标
app.post('/api/run', async (req, res) => {
    const start = Date.now();
    try {
        const { task, model, query } = req.body;
        const llmProvider = getLLMProvider(model);
        
        try {
            const response = await llmProvider.generate(query, "");
            
            res.json({
                task,
                model: llmProvider.name,
                modelApiKey: llmProvider.apiKey,
                query,
                rules: mockRules,
                original_response: "原始响应示例",
                enhanced_response: response,
                is_valid: true,
                violations: []
            });
        } catch (error) {
            console.error('API调用失败:', error);
            res.status(500).json({ error: error.message });
        }
    } catch (error) {
        errorRequests.inc();
        console.error('API调用失败:', error);
        res.status(500).json({ error: error.message });
    }
});

// 启动服务器
app.listen(PORT, '0.0.0.0', () => {
    console.log(`RuAG PoC 服务器运行在 http://0.0.0.0:${PORT}`);
});