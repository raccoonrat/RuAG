/**
 * MCTS规则学习过程可视化
 * 展示蒙特卡洛树搜索过程中的节点扩展和评估
 */
class MCTSVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;
        
        this.svg = d3.select(this.container).append('svg')
            .attr('width', '100%')
            .attr('height', '100%');
            
        this.simulation = null;
        this.nodes = [];
        this.links = [];
    }
    
    /**
     * 初始化MCTS树可视化
     */
    initTree() {
        // 创建根节点
        this.nodes = [{
            id: 0,
            name: "根节点",
            visits: 0,
            value: 0,
            depth: 0,
            expanded: false,
            selected: false
        }];
        
        this.links = [];
        
        // 创建力导向图
        this.simulation = d3.forceSimulation(this.nodes)
            .force('link', d3.forceLink(this.links).id(d => d.id).distance(80))
            .force('charge', d3.forceManyBody().strength(-200))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .on('tick', () => this.updateVisualization());
            
        // 初始化可视化
        this.updateVisualization();
    }
    
    /**
     * 更新可视化
     */
    updateVisualization() {
        // 清空SVG
        this.svg.selectAll('*').remove();
        
        // 绘制连接线
        const link = this.svg.append('g')
            .selectAll('line')
            .data(this.links)
            .enter().append('line')
            .attr('stroke', '#999')
            .attr('stroke-width', d => Math.sqrt(d.value) + 1)
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        
        // 绘制节点
        const node = this.svg.append('g')
            .selectAll('circle')
            .data(this.nodes)
            .enter().append('circle')
            .attr('r', d => 5 + Math.sqrt(d.visits))
            .attr('fill', d => {
                if (d.selected) return '#4CAF50';
                if (d.expanded) return '#ff7f50';
                return '#4a6fa5';
            })
            .attr('stroke', '#fff')
            .attr('stroke-width', 1.5)
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
        
        // 添加节点标签
        const label = this.svg.append('g')
            .selectAll('text')
            .data(this.nodes)
            .enter().append('text')
            .attr('x', d => d.x)
            .attr('y', d => d.y - 10)
            .attr('text-anchor', 'middle')
            .text(d => d.name)
            .style('font-size', '10px')
            .style('fill', '#333');
            
        // 添加访问次数标签
        const visits = this.svg.append('g')
            .selectAll('text')
            .data(this.nodes)
            .enter().append('text')
            .attr('x', d => d.x)
            .attr('y', d => d.y + 20)
            .attr('text-anchor', 'middle')
            .text(d => `访问: ${d.visits}`)
            .style('font-size', '8px')
            .style('fill', '#666');
    }
    
    /**
     * 模拟MCTS搜索过程
     * @param {number} iterations 迭代次数
     */
    simulateSearch(iterations = 10) {
        let currentIteration = 0;
        
        const interval = setInterval(() => {
            if (currentIteration >= iterations) {
                clearInterval(interval);
                return;
            }
            
            // 1. 选择阶段
            const selectedPath = this.selectNode();
            
            // 2. 扩展阶段
            if (selectedPath.length > 0) {
                const lastNode = selectedPath[selectedPath.length - 1];
                if (!lastNode.expanded && lastNode.visits > 0) {
                    this.expandNode(lastNode);
                }
            }
            
            // 3. 模拟阶段
            const reward = Math.random();
            
            // 4. 回溯阶段
            this.backpropagate(selectedPath, reward);
            
            // 更新可视化
            this.updateVisualization();
            
            currentIteration++;
        }, 1000);
    }
    
    /**
     * 选择节点
     * @returns {Array} 选择的路径
     */
    selectNode() {
        const path = [];
        let currentNode = this.nodes[0];
        currentNode.selected = true;
        path.push(currentNode);
        
        // 找到所有子节点
        const getChildren = (nodeId) => {
            return this.links
                .filter(link => link.source.id === nodeId)
                .map(link => this.nodes.find(node => node.id === link.target.id));
        };
        
        // 使用UCB选择子节点
        const selectChild = (children, parentVisits) => {
            if (children.length === 0) return null;
            
            // 有未访问的子节点，随机选择一个
            const unvisitedChildren = children.filter(child => child.visits === 0);
            if (unvisitedChildren.length > 0) {
                return unvisitedChildren[Math.floor(Math.random() * unvisitedChildren.length)];
            }
            
            // 使用UCB公式选择
            let bestScore = -Infinity;
            let bestChild = null;
            
            for (const child of children) {
                const exploitation = child.value / child.visits;
                const exploration = Math.sqrt(2 * Math.log(parentVisits) / child.visits);
                const ucbScore = exploitation + exploration;
                
                if (ucbScore > bestScore) {
                    bestScore = ucbScore;
                    bestChild = child;
                }
            }
            
            return bestChild;
        };
        
        // 遍历树
        while (true) {
            const children = getChildren(currentNode.id);
            const nextNode = selectChild(children, currentNode.visits);
            
            if (nextNode === null) break;
            
            nextNode.selected = true;
            path.push(nextNode);
            currentNode = nextNode;
        }
        
        return path;
    }
    
    /**
     * 扩展节点
     * @param {Object} node 要扩展的节点
     */
    expandNode(node) {
        const numChildren = Math.floor(Math.random() * 3) + 1; // 1-3个子节点
        
        for (let i = 0; i < numChildren; i++) {
            const childId = this.nodes.length;
            const childNode = {
                id: childId,
                name: `节点${childId}`,
                visits: 0,
                value: 0,
                depth: node.depth + 1,
                expanded: false,
                selected: false
            };
            
            this.nodes.push(childNode);
            
            this.links.push({
                source: node.id,
                target: childId,
                value: 1
            });
        }
        
        node.expanded = true;
        
        // 更新模拟
        this.simulation.nodes(this.nodes);
        this.simulation.force('link').links(this.links);
        this.simulation.alpha(1).restart();
    }
    
    /**
     * 回溯更新
     * @param {Array} path 路径
     * @param {number} reward 奖励值
     */
    backpropagate(path, reward) {
        for (const node of path) {
            node.visits += 1;
            node.value += reward;
            node.selected = false;
        }
    }
}