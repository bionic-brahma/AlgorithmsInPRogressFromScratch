%===========================================================
%The program is created by Devendra kaithal
%For the purpose of academic work
%===========================================================


%variables declaration which are crucial to control algorithm
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%n= number of candidates on swarm or population size
n=15;
%i= number of iterations
i=15;
%varn = the dimension of the objective function
varn=2;
%c1 and c2= constants used in velocity update equation
c1=2.05;
c2=2.05;
%fitness array for particles in the swarm
fitness= zeros(n,1);
%x= particles locations nxvarn dimension array
x= zeros(n,varn);
%v= velocities nxvarn dimension array
v= zeros(n,varn);
%feasibleL and feasibleU array will have 
%the feasible region for the variable used in the
%problem. L denotes lower bound and U denotes the upper bound
feasible= zeros(varn,2);
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%Initialization
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

feasible(1,1)=0;		
feasible(1,2)=7;
feasible(2,1)=1;
feasible(2,2)=15;

%random initialization of swarm
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for swarmid= 1:n
    for varindex= 1:varn
        x(swarmid,varindex)= feasible(varindex,1)+rand()*(feasible(varindex,2)-feasible(varindex,1));
        v(swarmid,varindex)= 0.0001*rand();

    end
end
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%calculation of objective function
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for swarmid= 1:n

    %In maximization, fitness is same as objective function
    %In minimization, reciprocal of (1+ objective function) can be taken

    fprintf('\nx values:  %d  %d ', x(swarmid,1),x(swarmid,2));
    %objecfun represent the obejective fubction
    fitness(swarmid,1)= (objecfun(x(swarmid,:)));
    fprintf('\n\tfitness : %d', fitness(swarmid,1));
end




[gbest,gbestx]= max(fitness);
gbestx= x(gbestx,:);
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%Calculation of pbest for the particles
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%For first iteration it is same as the fitness of the function at particle location
pbest=fitness;
pbestx=x;
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%initial steps are completed now, lets move to iteration
%velocity and position will be calculated and the c=2 has been taken

%Generation loop starts here
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%gen= the current generation number, i is the last generation number

for gen= 2:i
	fprintf('\n##########################################################' );
    
	%Particle update loop starts here
	%++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for swarmid= 1:n
		fprintf('\nswarmid: %d',swarmid);
            	fprintf('\n++++++++++++++++++++++++++++++++++++++++');
		%++++++++++++++++++++++++++++++++++++++++++++++
           	
		%Particles are getting updated vairiable by variable. Loop starts
	   	%++++++++++++++++++++++++++++++++++++++++++++++
		for varindex= 1:varn
                		fprintf('\nx%d:',varindex);
                		phai= c1+c2;                		
			%Chai calculation from Clerc and Kennedy [5]
                		chai= 2/(abs(2-phai-sqrt(phai*phai-4*phai)));
			%Velocity update equation

                		v(swarmid,varindex)=chai*(v(swarmid,varindex)+(c1*rand()*(pbestx(swarmid,varindex)-x(swarmid,varindex))+ c2*rand()*(gbestx(varindex)-x(swarmid,varindex))));

			%Position update equation
                     	x(swarmid,varindex)= x(swarmid,varindex)+ v(swarmid,varindex);

			%velocity clamping introduced 
			%combined with particle relocation

                 	if(x(swarmid,varindex)>feasible(varindex,2))
                        x(swarmid,varindex)=feasible(varindex,1)+rand()*(feasible(varindex,2)-feasible(varindex,1));
                 	end
                 	if(x(swarmid,varindex)<feasible(varindex,1))
		                 x(swarmid,varindex)=feasible(varindex,1)+rand()*(feasible(varindex,2)-feasible(varindex,1));
                	end
			%Display of some information
                		fprintf('\t pbest = %d', pbestx(swarmid,varindex));
                		fprintf('\tnew x%d = %d',varindex, x(swarmid,varindex));
                		fprintf('\t gbest = %d', gbestx(varindex));
                		fprintf('\t Velocity = %d', v(swarmid,varindex));
		end
		%++++++++++++++++++++++++++++++++++++++++++++++++
		%particles update loop ends here

		%Generating a particle plot for population
		plot(x(swarmid,1),x(swarmid,2),'*');
            	hold on;
            	fitness(swarmid,1)= objecfun(x(swarmid,:));
            	fprintf('\n\tfitness : %d', fitness(swarmid,1));
		%Updating Pbest
            	if pbest(swarmid,1) <= fitness(swarmid,1)
                		pbest(swarmid,1)= fitness(swarmid,1);
                		pbestx(swarmid,:) = x(swarmid,:);
            	end
        	end
	%Particle update loop ends here
    hold off;
    %updating Gbest
    if gbest< max(fitness)
        [gbest,gbestx]= max(fitness);
        gbestx= x(gbestx,:);
    end
    %Displaying best result of the genearation 
    fprintf('\nx values:  %d  %d  %d', gbestx(1),(gbestx(2)));
    fprintf('\ngbest : %d', gbest );
    %Waiting for key press to move to next generation
    w = waitforbuttonpress;
end
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%Program is ended here.
