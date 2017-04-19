function [] = savetxtdata(epoch,RL,CL,cur_path,method)
%quickly get pur , acc and nmi value,then save them in '.txt' format
cluster_nmi = nmi(CL',RL');%get NMI
cluster_pur = pur(CL',RL');%get PUR
cluster_acc = acc(RL,CL);%get ACC
cluster_results_path=strcat(cur_path);
if epoch==1
    fid=fopen(cluster_results_path,'wt');
    fprintf(fid,'%s\t','epoch');
    fprintf(fid,'%s\t','NMI');   
    fprintf(fid,'%s\t','PUR');
    fprintf(fid,'%s\n','ACC');
    fclose(fid);
end
fid=fopen(cluster_results_path,'at+');
fprintf(fid,'%g\t',epoch);
fprintf(fid,'%g\t',cluster_nmi);
fprintf(fid,'%g\t',cluster_pur);
fprintf(fid,'%g\n',cluster_acc);
fclose(fid);
end