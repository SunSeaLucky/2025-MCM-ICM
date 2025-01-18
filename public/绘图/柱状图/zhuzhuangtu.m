% 柱状图绘制模板
% By：阿昆的科研日常

%% 数据准备
% 读取数据
% dataset为5*3的矩阵，一行3个为一组，共5组
dataset = [0.241,0.393,0.294;
           0.219,0.254,0.238;
           0.238,0.262,0.272;
           0.198,0.329,0.287;
           0.201,0.197,0.185]; 

%% 颜色定义
% addcolor函数获取方式：
% 公众号后台回复：morecolor
C1 = addcolorplus(185);
C2 = 'k';
C3 = addcolorplus(178);

%% 图片尺寸设置（单位：厘米）
figureUnits = 'centimeters';
figureWidth = 12;
figureHeight = 8;

%% 柱状图绘制
%窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]); % define the new figure dimensions
hold on

% 绘制柱图
% 1-调节柱间距
GO = bar(dataset,1,'EdgeColor','k');

% 赋色
GO(1).FaceColor = C1;
GO(2).FaceColor = C2;
GO(3).FaceColor = C3;

% 文字注释，不需要可删
for ii=1:5
    text(ii-0.24,dataset(ii,1)+0.005,num2str(dataset(ii,1)),...
         'ROtation',90,'color',C1,'FontSize',10,'FontName',  'Helvetica');
    text(ii,dataset(ii,2)+0.01,num2str(dataset(ii,2)),...
         'ROtation',90,'color',C2,'FontSize',10,'FontName',  'Helvetica');     
    text(ii+0.22,dataset(ii,3)+0.01,num2str(dataset(ii,3)),...
         'ROtation',90,'color',C3,'FontSize',10,'FontName',  'Helvetica');  
end

% 坐标区调整
set(gca, 'Box', 'off', ...                                         % 边框
         'XGrid', 'off', 'YGrid', 'on', ...                        % 网格
         'TickDir', 'out', 'TickLength', [.01 .01], ...            % 刻度
         'XMinorTick', 'off', 'YMinorTick', 'off', ...             % 小刻度
         'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1],...           % 坐标轴颜色
         'YTick', 0:0.1:1,...                                      % 刻度位置、间隔
         'Ylim' , [0 0.6], ...                                     % 坐标轴范围
         'Xticklabel',{'samp1' 'samp2' 'samp3' 'samp4' 'samp5'},...% X坐标轴刻度标签
         'Yticklabel',{[0:0.1:1]})                                 % Y坐标轴刻度标签

% 标签及Legend 设置    
hYLabel = ylabel('RMSE (m)');
hLegend = legend([GO(1),GO(2),GO(3)], ...
                 'A', 'B', 'C', ...
                 'Location', 'northeast');
% Legend位置微调 
P = hLegend.Position;
hLegend.Position = P + [0.015 0.03 0 0];

% 刻度标签字体和字号
set(gca, 'FontName', 'Helvetica', 'FontSize', 9)
% 标签及Legend的字体字号 
set([hYLabel,hLegend], 'FontName',  'Helvetica')
set([hYLabel,hLegend], 'FontSize', 10)

% 背景颜色
set(gca,'Color',[1 1 1])

%% 图片输出
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'eg';
print(figureHandle,[fileout,'.png'],'-r300','-dpng');