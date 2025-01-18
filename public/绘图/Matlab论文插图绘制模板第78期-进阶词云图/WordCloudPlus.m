% 进阶词云图绘制模板
% 公众号：阿昆的科研日常

%% 数据准备
% 读取数据
% 此数据为Matlab自带
load sonnetsTable

%% 颜色定义
% TheColor函数获取方式：
% 公众号后台回复：TC
C = TheColor('sci',2068);
% 统计单词数并生成颜色变量
numWords = height(tbl);
r = randi([1,size(C,1)],numWords,1);
colors = C(r,1:3);

%% 图片尺寸设置（单位：厘米）
figureUnits = 'centimeters';
figureWidth = 16;
figureHeight = 10;

%% 窗口设置
figureHandle = figure('color','w');
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]); 

%% 进阶词云图绘制
wc = wordcloud(tbl,'Word','Count',...       % 将单词和相应的单词大小分别指定为Word和Count变量
                   'Color',colors,...       % 将单词颜色设置为C中随机值
                   'FontName','Arial',...   % 修改单词字体
                   'Shape','rectangle',...  % 修改词云形状
                   'LayoutNum',4,...        % 修改单词布局(非负整数)
                   'MaxDisplayWords',100,...% 要显示的最大单词数
                   'SizePower',1.2);        % 对尺寸应用的幂(正标量)，即尺寸.^ SizePower
title('');

%% 图片输出
print('test.png','-r300','-dpng');