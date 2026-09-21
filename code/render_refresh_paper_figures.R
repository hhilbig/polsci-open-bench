# Render the compact homepage figures with the manuscript's plotting theme.
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(tidyr);library(jsonlite);library(haschaR)})
set.seed(20260910)
args <- commandArgs(trailingOnly=TRUE)
root <- if(length(args)) args[1] else 'output/sidecar/refresh_20260910_release'
out <- file.path(root,'preview/llm-benchmark/figures')
manifest <- fromJSON(file.path(out,'figure-data.json'),simplifyVector=FALSE)
release <- fromJSON(file.path(root,'release.json'))
selected <- unlist(manifest$featured_models)
models <- release$models |> filter(model %in% selected) |> arrange(desc(mean_task_f1))
stopifnot(length(selected)>0,!anyDuplicated(selected),setequal(models$model,selected))
model_count <- nrow(models)
labels <- c('claude-opus-5'='Claude Opus 5','claude-sonnet-5'='Claude Sonnet 5',
 'gpt-6-astra'='GPT-6 Astra','gpt-5.6-sol'='GPT-5.6 Sol','gpt-5.6-terra'='GPT-5.6 Terra','gpt-5.6-luna'='GPT-5.6 Luna',
 'deepseek-v4-flash'='DeepSeek V4 Flash','deepseek-v4-pro'='DeepSeek V4 Pro',
 'jev-1.13.0'='Jev 1.13',
 'qwen3_8_27b_fp8'='Qwen3.8 27B','qwen3_8_flash_next_fp8'='Qwen3.8 Flash-Next*',
 'qwen3_6_27b_fp8'='Qwen3.6 27B','gemma4_31b_it_qat_w4a16'='Gemma 4 31B',
 'mistral_small_4_119b_nvfp4'='Mistral Small 4',
 'llama3_1_70b_instruct_fp8_dynamic_full34'='Llama 3.1 70B',
 'llama3_3_70b_instruct_fp8_dynamic'='Llama 3.3 70B')
colors <- c('claude-opus-5'='#999999','claude-sonnet-5'='#b0b0b0','gpt-6-astra'='#1f1f1f',
 'gpt-5.6-sol'='#333333','gpt-5.6-terra'='#666666','gpt-5.6-luna'='#8a8a8a',
 'deepseek-v4-flash'='#707070','deepseek-v4-pro'='#555555','jev-1.13.0'='#444444','qwen3_8_27b_fp8'='#d95f02',
 'qwen3_8_flash_next_fp8'='#7570b3','qwen3_6_27b_fp8'='#e6ab02','gemma4_31b_it_qat_w4a16'='#0072B2',
 'mistral_small_4_119b_nvfp4'='#e7298a',
 'llama3_1_70b_instruct_fp8_dynamic_full34'='#009E73',
 'llama3_3_70b_instruct_fp8_dynamic'='#1d6e52')
stopifnot(all(selected %in% names(labels)),all(selected %in% names(colors)))
meta <- bind_rows(lapply(manifest$task_metadata,function(x)x[c('task','paper_family','complexity')]))
stopifnot(nrow(meta)==34,!anyDuplicated(meta$task))
scores <- release$tasks |> filter(model %in% selected) |> select(model,task,headline_f1) |> left_join(meta,by='task',relationship='many-to-one')
stopifnot(nrow(scores)==model_count*34,!anyNA(scores$headline_f1),!anyDuplicated(scores[c('model','task')]))
models$model <- factor(models$model,levels=models$model)
scores$model <- factor(scores$model,levels=levels(models$model))
entries <- list()
save_plot <- function(p,stem,caption,values,w,h) {
 for(ext in c('svg','pdf','png')) ggsave(file.path(out,paste0(stem,'.',ext)),p,width=w,height=h,dpi=300,bg='white')
 entries[[length(entries)+1]] <<- list(file=stem,caption=caption,data=values)
}
p <- ggplot()+geom_jitter(data=scores,aes(model,headline_f1),width=.22,height=0,size=1.45,alpha=.18,color='grey78')+
 geom_point(data=models,aes(model,mean_task_f1,fill=model),shape=21,size=5,stroke=.6,color='grey20')+
 geom_text(data=models,aes(model,mean_task_f1,label=sprintf('%.3f',mean_task_f1)),vjust=-1.5,size=2.8,fontface='bold')+
 scale_fill_manual(values=colors,guide='none')+scale_x_discrete(labels=labels)+
 scale_y_continuous(limits=c(0,1.02),breaks=seq(0,1,.2))+
 labs(x=NULL,y='Mean F1 across tasks')+theme_hanno(fontsize=10.5)+
 theme(axis.text.x=element_text(angle=25,hjust=1),plot.margin=margin(5.5,12,12,16))
save_plot(p,'fig-recent-mean-f1',sprintf('Large circles show equal-task mean F1; faint dots show all 34 task scores. The %s models use the same 3,400 texts. The selection includes recent models and reference baselines, including the strong completed Llama checkpoints. *Flash-Next uses two GPUs. Hardware, quantization details and uncertainty intervals are in the model comparison.',model_count),models |> transmute(model=as.character(model),mean_f1=mean_task_f1),max(8.5,.65*model_count),4.3)
families <- c('Relevance & Harm','Position & Tone','Events & Actions','Claims & Relations','Issues & Topics')
questions <- c('Is it relevant or harmful?','What position or tone does it express?','What action or event is described?','What claim or relationship is asserted?','What issue is this about?')
facet_labels <- setNames(paste0(families,'\n"',questions,'"'),families)
# Paper's family panels group open models before API models, rather than re-ranking each panel.
family_order <- c(selected[!grepl('^(gpt-|claude-|deepseek-v4|jev-)',selected)],selected[grepl('^(gpt-|claude-|deepseek-v4|jev-)',selected)])
family <- scores |> group_by(paper_family,model) |> summarise(mean_f1=mean(headline_f1),.groups='drop') |>
 mutate(model=factor(model,levels=rev(family_order)),paper_family=factor(paper_family,levels=families))
p <- ggplot(family,aes(model,mean_f1,fill=model))+geom_col(width=.75)+geom_text(aes(label=sprintf('%.2f',mean_f1)),hjust=-.15,size=2.5,color='grey20')+
 coord_flip()+facet_wrap(~paper_family,ncol=2,labeller=labeller(paper_family=facet_labels))+
 scale_x_discrete(labels=labels)+scale_y_continuous(limits=c(0,1.05),expand=expansion(mult=c(0,.05)))+
 scale_fill_manual(values=colors,guide='none')+labs(x=NULL,y='Mean F1 within type')+theme_hanno(fontsize=10.5)+
 theme(axis.text.x=element_blank(),axis.ticks.x=element_blank(),strip.background=element_rect(fill='grey94',color='grey45',linewidth=.35),strip.text=element_text(face='bold',size=8))
save_plot(p,'fig-recent-family',sprintf('Mean F1 for the %s selected models within the paper’s original five annotation types, weighting tasks equally. *Flash-Next uses two GPUs. These task groupings differ from the category selector.',model_count),family |> transmute(model=as.character(model),family=as.character(paper_family),mean_f1),8.5,max(8.2,.60*model_count))
hardware <- setNames(models$provenance$hardware_tier,as.character(models$model))
comparison <- scores |> filter(hardware[as.character(model)] %in% c('api','single-gpu')) |>
 mutate(group=if_else(hardware[as.character(model)]=='api','API','Open (single GPU)'))
api_count <- sum(hardware=='api')
open_count <- sum(hardware=='single-gpu')
best <- comparison |> group_by(task,paper_family,complexity,group) |> summarise(f1=max(headline_f1),.groups='drop')
gap <- best |> select(task,group,f1) |> pivot_wider(names_from=group,values_from=f1) |>
 mutate(gap=API-`Open (single GPU)`,advantage=if_else(gap<0,'Open (single GPU)','API'))
p <- ggplot(gap,aes(y=reorder(task,gap)))+geom_vline(xintercept=0,color='grey45',linewidth=.45,linetype='dashed')+
 geom_segment(aes(x=0,xend=gap,yend=reorder(task,gap),color=advantage),linewidth=.5)+
 geom_point(aes(x=gap,fill=advantage),shape=21,size=2.8,stroke=.4,color='grey20')+
 scale_y_discrete(labels=function(x)gsub('_',' ',x))+scale_fill_manual(values=c('API'='#333333','Open (single GPU)'='#009E73'),name=NULL)+
 scale_color_manual(values=c('API'='#333333','Open (single GPU)'='#009E73'),guide='none')+
 labs(x='Best API minus best single-GPU open F1',y=NULL)+theme_hanno(fontsize=10)+theme(legend.position='bottom')
save_plot(p,'fig-recent-task-gap',sprintf('Best API minus best single-GPU open F1 on each task, using %s APIs and %s single-GPU open models in the default selection. Negative values favor open models. Winners are selected after observing results; this is descriptive, not a preselected-model comparison. Multi-GPU models are excluded.',api_count,open_count),gap |> select(task,gap),7.5,8.5)
complexity <- bind_rows(comparison |> group_by(complexity,group) |> summarise(mean_f1=mean(headline_f1),.groups='drop') |> mutate(panel='All model-task results'),
 best |> group_by(complexity,group) |> summarise(mean_f1=mean(f1),.groups='drop') |> mutate(panel='Best model per task in each class')) |>
 mutate(complexity=factor(complexity,levels=c('Low','Medium','High')))
p <- ggplot(complexity,aes(complexity,mean_f1,color=group,group=group))+geom_line(linewidth=.6)+geom_point(size=3)+
 facet_wrap(~panel,ncol=2)+scale_color_manual(values=c('API'='#333333','Open (single GPU)'='#009E73'),name=NULL)+
 scale_y_continuous(limits=c(0,1))+labs(x='Coding complexity',y='Mean F1')+theme_hanno(fontsize=10)+
 theme(legend.position='bottom',strip.background=element_rect(fill='grey94',color='grey45',linewidth=.35),strip.text=element_text(face='bold',size=8))
save_plot(p,'fig-recent-complexity',sprintf('F1 by the paper’s complexity rule, using the same %s APIs and %s single-GPU open models as the task-gap plot. High: multi-label output or at least eight effective labels. Medium: at least three effective labels or a prompt of at least 300 words. Otherwise low. Effective labels equal exp(entropy) of the frozen gold outcomes. Left averages all model-task scores; right averages the best score per task in each model class. Lines connect descriptive means.',api_count,open_count),complexity,8.5,3.5)
manifest$featured_figures <- entries
write_json(manifest,file.path(out,'figure-data.json'),auto_unbox=TRUE,pretty=TRUE,digits=NA,null='null')
