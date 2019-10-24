#include <iostream.h>
#include<conio.h>
#include<dos.h>
#include<stdlib.h>
#include<ctype.h>

void QuickSort(int[],int,int);                 //  pack jo extra rkhni thi choice ke
int Partition(int[],int,int);                  //  liye wo isme include nhi ki gyi hai.
int throwcard(int[]);                          //  but krna jyada muskil nhi hai.


 void main(){
 int putthecard, user_entry=-1;
 int beg,end;

 int mycards[13]={1,7,16,19,25,29,35,40,45,56,78,88,97}; // ye 13 cards jo program ko diye jaaenge.

 clrscr();
 while(user_entry!=0){
 putthecard=throwcard(mycards);
 cout<<"\nMY TURN:\n"<<mycards[putthecard];       // this card is thrown by programm.
 cout<<"\nUSER TURN: place the card\n" ;
 cin>>user_entry;
			 //check whether this card is available to user or not;
 mycards[putthecard]=user_entry;
 beg=0;
 end=12;
 QuickSort(mycards,beg,end);
 cout<<"\nAfter Sorting : \n";
 for(int i=0;i<13;i++){
 cout<<mycards[i]<<" ";
 }
 }
 getch();
 }

int throwcard(int num[13]){
  int ps[13],pe[13],play[13];
  int l=0;
 // cout<<"\n=====================================\n";
  int s1,s2=0,q=-1;;
  for(int j=1;j<13;j++){
  s1=j-1;
  while((num[j]-num[j-1]<3)&&j<13){              // here 2 is the maximum difference that the two consequtive cards can have.
 // delay(50);
  //cout<< " j="<<j-1;
  //cout<<" ";
 // cout<<num[j-1];
  s2=j;
  j++;
  }
  if(s2-s1>=2){                // here s2-s1 decides minimum how many cards can be in group
  ps[l]=s1;        // return for game
  pe[l]=s2;        // return for game
 // cout<<"\nfrom inside initializer l="<<l;
  l++;  }          // return for game

  ///// arry set take positions
 // cout<<"\ns1:"<<s1<<" s2:"<<s2<<"\n";
  if(s1>=s2){
  q++;             //return for game       // check for =
  play[q]=s1;     //return for game
  }
  }
 // getch();
  l--;
 // cout<<"\nGroups(not for play):"   ;
  while(l!=-1){
 // cout<<"\n===============\nl="<<l<<"  "<<ps[l]<<"--"<<pe[l]<<"\n==============\n";
  cout<<"\n";
  if(pe[l]-ps[l]>=3){
  cout<<"\n\t\tI am WINNER with cards\n";
  for(int k=ps[l];k<=pe[l];k++)
  cout<<" "<<num[k];
  getch();
  exit(1);
  }
  l--;
  }
 // getch();
  randomize();
  int thro= random(q+1);
 // cout<<"\nplay cards: \n";
  while(q!=-1) {
//  cout<<" "<<num[play[q]]   ;
  q--;}
 // cout<<"\nthrown card: "<<num[play[thro]];  // return this throw card to main.
  getch();
  return play[thro]; // returning the position of card in real array.
  }






////sorting algo/////////////////////////////////

void QuickSort(int a[], int beg, int end){
if(beg<end){
int p=Partition(a,beg,end);
QuickSort(a,beg,p-1);
QuickSort(a,p+1,end);
}
}

int Partition(int a[], int beg, int end)  { // fuction for pivot
int p=beg, pivot=a[beg], loc;
for(loc=beg+1;loc<=end;loc++){
if(pivot>a[loc]){
a[p]=a[loc];
a[loc]=a[p+1];
a[p+1]=pivot;
p=p+1;
}
}
return p;
}