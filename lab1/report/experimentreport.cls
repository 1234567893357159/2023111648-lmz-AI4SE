\documentclass{experimentreport}

\input{listing_style.tex}
\usepackage{graphicx}
\usepackage{caption}
%========================
% 基本信息
%========================

\setcoursename{基础软件开发技术实践}
\setreportname{智能软件工程实验报告}

\setexperimenttitle{\hintholder{代码审查意见挖掘与需求理解}}

\setstudentname{\hintholder{刘明卓}}
\setdepartment{\hintholder{计算学部}}
\setstudentid{\hintholder{2023111648}}

\setcourseteacher{刘佳琨}
\setinstructor{刘佳琨}

\setlablocation{\hintholder{格物楼213}}
\setlabtime{    \hintholder{2026.6.29}}

\begin{document}

%========================
% 自动生成封面
%========================

\makereporthead

%========================
% 实验目的
%========================

\experimentpurpose{

完成本实验后，学生应能够：
\begin{enumerate}
    \item 理解现代软件开发过程中PullRequest（PR）与 Code Review 的基本流程。
    \item 熟悉GitHub 开源软件仓库中代码审查数据的组织形式。
    \item 掌握Pull Request、Commit、Review、Comment 等对象之间的关系。
    \item 能够利用GitHub API 或公开数据集获取代码审查数据。
    \item 能够完成代码、文本及元数据等多源信息的抽取。
    \item 掌握数据集统计分析方法，为后续模型训练提供数据基础。
\end{enumerate}

}

%========================
% 实验内容
%========================

\experimentcontent{
\noindent\textbf{1.7 实验步骤}
\begin{enumerate}
    \item \textbf{步骤一：获取 Pull Request}
    本次实验选取5个大型开源软件项目，调用GitHub API为每个项目采集300条PR数据，不强制单条PR同时包含AI生成代码与AI审查记录。采集字段包含PRID、Title、Body、Author、Created Time、Merge Status。

    \item \textbf{步骤二：获取 Code Review}
    基于已获取的PR数据集，继续通过GitHub API拉取每条PR对应的代码审查信息，提取字段：Reviewer、Review Decision、Review Time、Review Comment。

    \item \textbf{步骤三：获取代码修改}
    提取每条PR的代码变更细节，包含修改文件、修改函数、Diff代码差异、Commit Message提交说明。

    \item \textbf{步骤四：提取特征}
    整合前序全部数据完成多维度特征统计，统计指标包括：PR长度、修改文件数、Reviewer数量、Comment数量、是否Merge、Label数量、是否包含AIReviewer、是否存在AIGenerated Code。

    \item \textbf{步骤五：数据分析}
    对特征数据集开展统计分析，内容包括合并与未合并PR数量统计、Review Comment数量分布、Label分布、Reviewer数量分布、PR长度分布、AI相关PR与人工PR数量对比；绘制柱状图、饼图、直方图完成数据可视化展示。
\end{enumerate}
}

%========================
% 实验结果
%========================

\experimentresult{

本实验通过 GitHub API 成功获取了 5 个大型开源项目共计 1500 条 Pull Request 数据，提取了 36 个维度的特征，并完成了系统的统计分析与可视化。

\vspace{0.3cm}
\noindent\textbf{1. 数据集概览}

本实验采集的数据集包含：
\begin{itemize}
    \item 总 PR 数量：1500 条（每个项目 300 条）
    \item 特征维度：36 维（基础特征 16 个、Review 特征 12 个、AI 检测特征 5 个、语言特征 3 个）
    \item 覆盖项目：kubernetes/kubernetes、pytorch/pytorch、tensorflow/tensorflow、microsoft/vscode、langchain-ai/langchain
\end{itemize}

基本统计结果如下：
\begin{itemize}
    \item PR 平均长度：1851 字符，中位数 772 字符
    \item 平均修改文件数：10.73 个
    \item 平均 Reviewer 数量：0.82 人，中位数 0 人
    \item 平均评论总数：3.73 条，中位数 2 条
    \item 合并率：36.73\%（551 个合并，949 个未合并）
    \item AI Reviewer 参与率：74.13\%
    \item AI 生成代码占比：22.93\%
\end{itemize}

\vspace{0.3cm}
\noindent\textbf{2. Merge 与 Non-Merge 分布}

\begin{center}
\includegraphics[width=0.9\textwidth]{01_merge_vs_non_merge.png}
\end{center}

统计结果显示，未合并 PR 数量（949）多于合并 PR 数量（551），合并率约为 36.7\%。

\vspace{0.3cm}
\noindent\textbf{3. Review Comment 数量分布}

\begin{center}
\includegraphics[width=0.9\textwidth]{02_comment_distribution.png}
\end{center}

Comment 数量分布呈明显的右偏分布，均值为 3.73，中位数为 2，大多数 PR 评论数量较少。

\vspace{0.3cm}
\noindent\textbf{4. Label 分布}

\begin{center}
\includegraphics[width=0.8\textwidth]{03_label_distribution.png}
\end{center}

\texttt{cncf-cla: yes}、\texttt{needs-triage}、\texttt{needs-priority} 等自动化标签出现频率最高，反映了开源项目中自动化流程的广泛应用。

\vspace{0.3cm}
\noindent\textbf{5. Reviewer 数量分布}

\begin{center}
\includegraphics[width=0.8\textwidth]{04_reviewer_count_distribution.png}
\end{center}

超过 40\% 的 PR 没有人工 Reviewer，多由 CI Bot 自动检查。多数 PR 仅 1-2 个 Reviewer。

\vspace{0.3cm}
\noindent\textbf{6. PR 长度分布}

\begin{center}
\includegraphics[width=0.9\textwidth]{05_pr_length_distribution.png}
\end{center}

PR 长度分布范围从 0 到 22888 字符，对数尺度下近似正态分布。

\vspace{0.3cm}
\noindent\textbf{7. AI 与 Human 对比}

\begin{center}
\includegraphics[width=0.9\textwidth]{06_ai_vs_human.png}
\end{center}

分析结果表明：74.1\% 的 PR 有 AI/Bot Reviewer 参与，22.9\% 的 PR 包含 AI 生成代码。

\vspace{0.3cm}
\noindent\textbf{8. 各仓库指标对比}

\begin{center}
\includegraphics[width=0.9\textwidth]{07_per_repo_comparison.png}
\end{center}

不同项目在合并率、AI 使用率等指标上存在明显差异：langchain-ai/langchain 的 AI 生成代码占比最高，kubernetes 审查流程最严格。

\vspace{0.3cm}
\noindent\textbf{9. 特征相关性分析}

\begin{center}
\includegraphics[width=0.8\textwidth]{08_correlation_heatmap.png}
\end{center}

相关性分析发现：\texttt{files\_changed} 与 \texttt{additions}、\texttt{deletions} 强正相关；\texttt{reviewer\_count} 与 \texttt{total\_comment\_count} 中等正相关；AI Reviewer 与合并概率相关性较弱。

\vspace{0.3cm}
\noindent\textbf{10. 编程语言分布}

\begin{center}
\includegraphics[width=0.9\textwidth]{09_language_distribution.png}
\end{center}

编程语言分布反映了所选项目特性：Go（kubernetes）、Python（pytorch、langchain）、C++（pytorch、tensorflow）、TypeScript（vscode）占主导。

\vspace{0.3cm}
\noindent\textbf{实验结果总结}

本实验成功完成了预期目标：通过 GitHub API 获取了 1500 条 PR 数据，完成 36 维特征提取，通过 9 项统计分析和可视化揭示了开源项目 PR 数据的统计规律，验证了 AI 在现代软件开发中的广泛应用（近 23\% 的 PR 包含 AI 生成代码），构建的数据集可用于后续实验任务。

}

%========================
% 问题总结
%========================

\experimentproblems{

请总结实验中遇到的问题及解决方法

问题1：用github爬取pr的时候会出现ssl出错

解决1：改了ssl之后才发现用github加速会改我自己放出的请求，把加速关了就好了

问题2：有一些pr是机器自动生成的

解决2：直接在分一个类打一个标签就行，在之后的时候再决定是否把他当成人类代码
}

%========================
% 实验体会
%========================

\experimentexperience{

请分享实验体会和收获

通过本次实验，我掌握了GitHub API的使用和PR数据采集方法，学会了从多源数据中提取特征并进行统计分析。实践中，我体会到不同开源项目在审查文化上的显著差异，也看到AI/Bot在代码审查中的高参与度（超七成PR有AI审查）。此外，我认识到数据清洗与特征工程对后续建模的重要性，为智能软件工程研究积累了宝贵的实践经验。

}

%========================
% 教师评语
%========================

\teachercomment

\end{document}