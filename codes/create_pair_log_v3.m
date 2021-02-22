%create log file after lookup table
load('Normal_abnormal_v4/Net_list_8_12_20/im_list.mat');
load('Normal_abnormal_v4/Net_list_8_12_20/labels.mat');
load('Normal_abnormal_v4/Net_list_8_12_20/lookup.mat');

an_list=string(im_list);
name_list=string(erase(an_list,{'.jpg','\'}));
labels=string(labels);

new_test_list=[name_list labels];
% n=1;
% an=1;
% for i=1:length(tst_list)
%     num = find(strcmp(tst_list(i),an_list));
%     new_test_list(i,1) = tst_list(i);
%     if(strcmp(AN_list.AN_Type(num),"Normal"))
% %         normal_list(n,2)=AN_list.AN_Type(num);
% %         normal_list(n,1) = tst_list(i);
% %         n=n+1;
%         new_test_list(i,2) = AN_list.AN_Type(num);
%     else
% %         abnormal_list(an,2)="Abnormal";
% %         abnormal_list(an,1) = tst_list(i);
% %         an=an+1;
%         new_test_list(i,2) = "Abnormal";
%     end
% end
% % new_list_v2 = [normal_list;abnormal_list];
p=0;
flag=0;
check=[];

for i=1:length(new_test_list)
    if(flag>i)
        i=flag;
    end
    

    type=new_test_list(i,2);
    name=new_test_list(i,1);
    if(contains(name,'R'))
        pairs =find(contains(im_list,strrep(name,'R','L')));
    else if(contains(name,'L'))
            pairs =find(contains(im_list,strrep(name,'L','R')));
        end
    end
    
    
    if(pairs & ~any(check==pairs))
        pairs
        p=p+1
        
        if(contains(name,'R'))
            log(p,1)=name;
            log(p,2)=type;
            log(p,[5:6])=lookup(i,:);
            check=[check i];
%             i=i+1;
            
            pr=pairs(1);
            check=[check pr];
            type=new_test_list(pr,2);
            name=new_test_list(pr,1);
            log(p,3)=name;
            log(p,4)=type;
            log(p,[7:8])=lookup(pr,:);
%             flag=i+1;
        else if (contains(name,'L'))
            log(p,3)=name;
            log(p,4)=type;
            log(p,[7:8])=lookup(i,:);
            check=[check i];
%             i=i+1;
            
            pr=pairs(1);
            check=[check pr];
            type=new_test_list(pr,2);
            name=new_test_list(pr,1);
            log(p,1)=name;
            log(p,2)=type;
            log(p,[5:6])=lookup(pr,:);
%             flag=i+1;
            end
        end
    end
    
    
end
nl=[];
for i=1:length(log)
    if (log(i,2)=="Normal" & log(i,4)=="Normal" )
        nl=[nl; log(i,:)];
    end
end
al=[];
for i=1:length(log)
    if (log(i,2)=="Abnormal" & log(i,4)=="Abnormal" )
        al=[al; log(i,:)];
    end
end
nal=[];
for i=1:length(log)
    if (log(i,2)=="Normal" & log(i,4)=="Abnormal" )
        nal=[nal; log(i,:)];
    end
end
anl=[];
for i=1:length(log)
    if (log(i,2)=="Abnormal" & log(i,4)=="Normal" )
        anl=[anl; log(i,:)];
    end
end
new_log=[al;nl;anl;nal];