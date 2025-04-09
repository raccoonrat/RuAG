/**
 * 规则可视化组件
 * 用于展示规则的结构、应用过程和效果
 */
class RuleVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.svg = d3.select(this.container).append('svg')
            .attr('width', '100%')
            .attr('height', '100%');
    }
    
    /**
     * 可视化规则树
     * @param {Array} rules 规则数组
     */
    visualizeRuleTree(rules) {
        // 清空容器
        this.svg.selectAll('*').remove();
        
        // 设置树布局
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        // 创建层次结构数据
        const root = {
            name: "规则集",
            children: rules.map((rule, index) => ({
                name: `规则 ${index + 1}`,
                accuracy: rule.accuracy,
                children: [
                    { name: "条件", value: rule.condition },
                    { name: "预测", value: rule.prediction }
                ]
            }))
        };
        
        // 创建树布局
        const treeLayout = d3.tree().size([height - 40, width - 160]);
        
        // 创建层次结构
        const hierarchy = d3.hierarchy(root);
        
        // 计算节点位置
        const treeData = treeLayout(hierarchy);
        
        // 绘制连接线
        this.svg.selectAll('.link')
            .data(treeData.links())
            .enter().append('path')
            .attr('class', 'link')
            .attr('d', d => {
                return `M${d.source.y},${d.source.x}
                        C${(d.source.y + d.target.y) / 2},${d.source.x}
                         ${(d.source.y + d.target.y) / 2},${d.target.x}
                         ${d.target.y},${d.target.x}`;
            })
            .attr('fill', 'none')
            .attr('stroke', '#999')
            .attr('stroke-width', 1.5);
        
        // 创建节点组
        const node = this.svg.selectAll('.node')
            .data(treeData.descendants())
            .enter().append('g')
            .attr('class', d => `node ${d.children ? 'node--internal' : 'node--leaf'}`)
            .attr('transform', d => `translate(${d.y},${d.x})`);
        
        // 添加节点圆圈
        node.append('circle')
            .attr('r', d => d.data.accuracy ? 10 * d.data.accuracy : 5)
            .attr('fill', d => {
                if (d.depth === 0) return '#4a6fa5';
                if (d.depth === 1) return '#ff7f50';
                return '#ffd700';
            })
            .attr('stroke', '#fff')
            .attr('stroke-width', 2);
        
        // 添加节点文本
        node.append('text')
            .attr('dy', '.31em')
            .attr('x', d => d.children ? -8 : 8)
            .attr('text-anchor', d => d.children ? 'end' : 'start')
            .text(d => d.data.name)
            .style('font-size', '12px')
            .style('fill', '#333');
        
        // 为叶子节点添加值标签
        node.filter(d => !d.children && d.data.value)
            .append('text')
            .attr('dy', '1.5em')
            .attr('x', 8)
            .attr('text-anchor', 'start')
            .text(d => d.data.value)
            .style('font-size', '10px')
            .style('fill', '#666');
    }
    
    /**
     * 可视化规则应用过程
     * @param {string} text 输入文本
     * @param {Array} rules 应用的规则
     * @param {Array} matches 规则匹配结果
     */
    visualizeRuleApplication(text, rules, matches) {
        // 清空容器
        this.svg.selectAll('*').remove();
        
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        // 绘制输入文本
        this.svg.append('text')
            .attr('x', 10)
            .attr('y', 20)
            .text('输入: ' + text)
            .style('font-size', '14px')
            .style('font-weight', 'bold');
        
        // 绘制规则应用流程
        const flowG = this.svg.append('g')
            .attr('transform', `translate(10, 40)`);
        
        // 为每个匹配的规则创建一个流程项
        matches.forEach((match, i) => {
            const ruleG = flowG.append('g')
                .attr('transform', `translate(0, ${i * 60})`);
            
            // 规则框
            ruleG.append('rect')
                .attr('x', 0)
                .attr('y', 0)
                .attr('width', 150)
                .attr('height', 40)
                .attr('rx', 5)
                .attr('fill', '#f0f0f0')
                .attr('stroke', '#999');
            
            // 规则文本
            ruleG.append('text')
                .attr('x', 75)
                .attr('y', 25)
                .attr('text-anchor', 'middle')
                .text(`规则 ${match.ruleIndex + 1}`)
                .style('font-size', '12px');
            
            // 箭头
            ruleG.append('path')
                .attr('d', 'M150,20 L180,20')
                .attr('stroke', '#999')
                .attr('stroke-width', 2)
                .attr('marker-end', 'url(#arrow)');
            
            // 匹配结果框
            ruleG.append('rect')
                .attr('x', 180)
                .attr('y', 0)
                .attr('width', 200)
                .attr('height', 40)
                .attr('rx', 5)
                .attr('fill', match.matched ? '#d4edda' : '#f8d7da')
                .attr('stroke', match.matched ? '#c3e6cb' : '#f5c6cb');
            
            // 匹配结果文本
            ruleG.append('text')
                .attr('x', 280)
                .attr('y', 25)
                .attr('text-anchor', 'middle')
                .text(match.matched ? '匹配成功' : '匹配失败')
                .style('font-size', '12px');
            
            // 如果匹配成功，添加应用结果
            if (match.matched) {
                // 箭头
                ruleG.append('path')
                    .attr('d', 'M380,20 L410,20')
                    .attr('stroke', '#999')
                    .attr('stroke-width', 2)
                    .attr('marker-end', 'url(#arrow)');
                
                // 应用结果框
                ruleG.append('rect')
                    .attr('x', 410)
                    .attr('y', 0)
                    .attr('width', 200)
                    .attr('height', 40)
                    .attr('rx', 5)
                    .attr('fill', '#e2f0fb')
                    .attr('stroke', '#b8daff');
                
                // 应用结果文本
                ruleG.append('text')
                    .attr('x', 510)
                    .attr('y', 25)
                    .attr('text-anchor', 'middle')
                    .text(match.result)
                    .style('font-size', '12px');
            }
        });
        
        // 添加箭头标记定义
        this.svg.append('defs').append('marker')
            .attr('id', 'arrow')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 8)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', '#999');
    }
}