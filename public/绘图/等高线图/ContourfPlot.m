clear;
clc;

% Excel文件路径
excelFilePath = '深度数据.xlsx';

% 读取Excel文件
T = readmatrix(excelFilePath);

% 获取数据
xi = T(1,2:end); % 假设T的第一行是x坐标（除了第一个元素是标题外）
yi = T(2:end,1);

% 将表格数据转换为双精度数组以便进行数学运算
Z = -T(2:end,2:end); 

% 构建网格数据
[X,Y] = meshgrid(xi,yi);

%% 颜色定义
% 注意：addcolorplus函数的实现未给出，此处假设您已有此函数或会替换为MATLAB内置 colormap
map = addcolorplus(302);
% map = flipud(map);

%% 图片尺寸设置（单位：厘米）
figureUnits = 'centimeters';
figureWidth = 16;
figureHeight = 10;

%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);

%% 绘制等高线填充图
hContour = contourf(X, Y, Z, 10, 'LineWidth',1.2);
hTitle = title('Contourf Plot');
hXLabel = xlabel('XAxis');
hYLabel = ylabel('YAxis');

% 获取当前轴上的所有等高线对象
contourLines = findall(gca, 'Type', 'patch');

% 使用轮廓线的'FaceVertexCData'属性来获取等高线对应的值
for k = 1:length(contourLines)
    levels = contourLines(k).FaceVertexCData;
    uniqueLevels = unique(levels);
    for j = 1:length(uniqueLevels)
        idx = levels == uniqueLevels(j);
        xy = contourLines(k).Vertices(idx,:);
        if sum(idx) > 1 % 确保至少有两个点构成线段
            midPointIdx = floor(sum(idx)/2)+1; % 计算中间点索引
            labelPosition = mean(xy(idx,:),2); % 计算中间点位置
            text(labelPosition(1), labelPosition(2), num2str(round(uniqueLevels(j),2)), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'FontSize', 8, 'Color', 'black', 'Clipping', 'on');
        end
    end
end

%% 细节优化
colormap(map)
colorbar
axis equal
set(gca, 'Box', 'off', ...
         'LineWidth', 1,...
         'XGrid', 'off', 'YGrid', 'off', ...
         'TickDir', 'out', 'TickLength', [.015 .015], ...
         'XMinorTick', 'on', 'YMinorTick', 'on', ...
         'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1])
hold on
XL = get(gca,'xlim'); XR = XL(2);
YL = get(gca,'ylim'); YT = YL(2);
xc = get(gca,'XColor');
yc = get(gca,'YColor');
plot(XL,YT*ones(size(XL)),'color', xc,'LineWidth',1)
plot(XR*ones(size(YL)),YL,'color', yc,'LineWidth',1)
set(gca, 'FontName', 'Helvetica')
set([hXLabel, hYLabel], 'FontName', 'AvantGarde')
set(gca, 'FontSize', 10)
set([hXLabel, hYLabel], 'FontSize', 11)
set(hTitle, 'FontSize', 11, 'FontWeight' , 'bold')
set(gcf,'Color',[1 1 1])

%% 图片输出
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'test';
print(figureHandle,[fileout,'.png'],'-r300','-dpng');