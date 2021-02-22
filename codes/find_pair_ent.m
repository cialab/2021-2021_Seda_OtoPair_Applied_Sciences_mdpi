normal= dir('X:\Autoscope\Tube_Effusion_Normal - 11_7_19\Normal\*.jpg');
effusion= dir('X:\Autoscope\Tube_Effusion_Normal - 11_7_19\Effusion\*.jpg');
tube = dir('X:\Autoscope\Tube_Effusion_Normal - 11_7_19\Tube\*.jpg');

n = struct('name',{normal.name});
t = struct('name',{tube.name});
e = struct('name',{effusion.name});

no=struct2cell(n);
nor=string(no);
nor1(1,:)=nor(1,1,:);
normal_t(:,1)=nor1';
normal_t(:,2)=repmat('normal',[1]);


ef=struct2cell(e);
eff=string(ef);
eff1(1,:)=eff(1,1,:);
effusion_t(:,1) = eff1';
effusion_t(:,2) = repmat('effusion',[1]);

tu=struct2cell(t);
tub=string(tu);
tub1(1,:)=tub(1,1,:);
tube_t(:,1) = tub1';
tube_t(:,2) = repmat('tube',[1]);
n_e=0;
for i=1:length(nor1)
    name_n = char( nor1(1,i));
    for j=1:length(eff1)
        name_e = char(eff1(1,j));
        if(strcmp(name_n(1:end-5),name_e(1:end-5)))
            n_e=n_e+1
            n_e_list(n_e,[1 2])=normal_t(i,:);
            n_e_list(n_e,[3 4])=effusion_t(j,:);
        end
        %abc(i,:)= contains(eff1,name(1:end-5));
    end
end

n_t=0;
for i=1:length(nor1)
    name_n = char( nor1(1,i));
    for j=1:length(tub1)
        name_t = char(tub1(1,j));
        if(strcmp(name_n(1:end-5),name_t(1:end-5)))
            n_t=n_t+1;
            n_t_list(n_t,[1 2])=normal_t(i,:);
            n_t_list(n_t,[3 4])=tube_t(j,:);
        end
    end
end

e_t=0;
for i=1:length(eff1)
    name_e = char( eff1(1,i));
    for j=1:length(tub1)
        name_t = char(tub1(1,j));
        if(strcmp(name_e(1:end-5),name_t(1:end-5)))
            e_t=e_t+1
            e_t_list(e_t,[1 2])=effusion_t(i,:);
            e_t_list(e_t,[3 4])=tube_t(j,:);
        end
    end
end

n_n=0;
for i=1:length(nor1)
    name_n1 = char( nor1(1,i));
    for j=1:length(nor1)
        name_n2 = char(nor1(1,j));
        if(~(i==j) & (strcmp(name_n1(1:end-5),name_n2(1:end-5))))
            n_n=n_n+1
            n_n_list(n_n,[1 2])=normal_t(i,:);
            n_n_list(n_n,[3 4])=normal_t(j,:);
        end
    end
end
e_e=0;
for i=1:length(eff1)
    name_e1 = char( eff1(1,i));
    for j=1:length(eff1)
        name_e2 = char(eff1(1,j));
        if(~(i==j) & (strcmp(name_e1(1:end-5),name_e2(1:end-5))))
            e_e=e_e+1
            e_e_list(e_e,[1 2])=effusion_t(i,:);
            e_e_list(e_e,[3 4])=effusion_t(j,:);
        end
    end
end

t_t=0;
for i=1:length(tub1)
    name_t1 = char( tub1(1,i));
    for j=1:length(tub1)
        name_t2 = char(tub1(1,j));
        if(~(i==j) & (strcmp(name_t1(1:end-5),name_t2(1:end-5))))
            t_t=t_t+1
            t_t_list(t_t,[1 2])=tube_t(i,:);
            t_t_list(t_t,[3 4])=tube_t(j,:);
        end
    end
end
% first_name = blanks(15);
% second_name = blanks(15);
% k=1;
% j=1;
% pair_count =0;
% same_an_pair=0;
% same_pair_count = 0;
% for i=1:sz-1
%     
%     first_name = NewFormat121318{i,4};
%     first = char(first_name);
%     second_name = NewFormat121318{i+1,4};
%     second = char(second_name);
%     l1 = length(first);
%     l2 = length(second);
%     if(strcmp(first(1:l1-1),second(1:l2-1)))
%         pair_count = pair_count+1;
%         T_pair(j,1)=NewFormat121318(i,4);
%         T_pair(j,2)=NewFormat121318(i,9);
%         T_pair(j,3)=NewFormat121318(i,5);
%         T_pair(j+1,1)=NewFormat121318(i+1,4);  
%         T_pair(j+1,2)=NewFormat121318(i+1,9); 
%         T_pair(j+1,3)=NewFormat121318(i+1,5);
%         
%         All_p(pair_count,1:2) = T_pair(j,1:2);
%         All_p(pair_count,3:4) = T_pair(j+1,1:2);
%         
%         Type_list(k,1)=table2cell(NewFormat121318(i,9));
%         Type_list(k,2)=table2cell(NewFormat121318(i+1,9));
%         
%         if(NewFormat121318{i,9}==NewFormat121318{i+1,9})
%             same_pair_count =same_pair_count+1;
%             same_an_pair = same_an_pair +2;
%             Same_pair(same_an_pair,1)=NewFormat121318(i,4);
%             Same_pair(same_an_pair,2)=NewFormat121318(i,9);
%             Same_pair(same_an_pair,3)=NewFormat121318(i,5);
%             Same_pair(same_an_pair+1,1)=NewFormat121318(i+1,4);  
%             Same_pair(same_an_pair+1,2)=NewFormat121318(i+1,9);  
%             Same_pair(same_an_pair+1,3)=NewFormat121318(i+1,5); 
%             Same_p(same_pair_count,1:2) = Same_pair(same_an_pair,1:2);
%             Same_p(same_pair_count,3:4) = Same_pair(same_an_pair+1,1:2);
%         end
%         k=k+1;
%         i=i+1;
%         j=j+2;
%         
%     end
% end
