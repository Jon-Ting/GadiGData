!=======================================================================
!   NCPac 
!
!   Contact:  
!   Dr George Opletal 
!   (george.opletal@data61.csiro.au / g.opletal@gmail.com)
!
!=======================================================================


!=======================================================================
!START OF KINDS MODULE    

MODULE KINDS

IMPLICIT NONE

integer, parameter :: sp=kind(1.0)                                      !single precision definition
integer, parameter :: dp=kind(1.0d0)                                    !double precision definition

END MODULE KINDS

!END OF KINDS MODULE
!=======================================================================



!=======================================================================
!START OF VARIABLES MODULE    

MODULE VARIABLES

    USE KINDS
    IMPLICIT NONE

    !Version 
    character(len=100) :: con_vers = '1.0.0'                            !code version number

    
    !Program constants    
    integer, parameter :: con_maxnn = 1200                              !maximum number of 1st nearest neighbors    
    integer, parameter :: con_maxtypes = 10                             !maximum number of particle types    

    !Physical constants
    real(dp), parameter :: con_pi = 3.1415926589793_dp    
    
    
    
    !Input parameters(in order of appearance in input file)  
    character(len=100) :: in_filexyz                                    !input xyz movie filename
    integer :: in_xyz_prec                                              !number of decimal places in coordinates
    integer :: in_write_screen                                          !write out to screen flag
    integer :: in_5th_option                                            !option for reading in 4th surface column
    real(dp) :: in_cutoff(con_maxtypes,con_maxtypes)                    !cutoff distance for defining 1st nn   
    
    real(dp) :: in_xl, in_yl, in_zl                                     !cell length in x, y and z direction     
    
    integer :: in_frames_start, in_frames_end, in_frames_jump           !starting frame, end frame, no. of frames between analysis
    
    real(dp) :: in_density                                              !reduced density
    real(dp) :: in_delr                                                 !spacing of g(r) functions 
    integer :: in_gr_points                                             !number of points in g(r)
    integer :: in_sq_points                                             !number of points in S(q)
    
    integer :: in_option_cluster                                        !flag for using cluster filter  
    integer :: in_mincluster                                            !cluster size minimum below which removal
    
    integer :: in_option_surf                                           !flag for finding surface routines          
    real(dp) :: in_cone_angle                                           !cone angle beyond which defines surface atoms                                     
    integer :: in_surf_points                                           !points in uniform point distribution
    
    integer :: in_q6order_flag                                          !q6 order analysis flag for on and off
    real(dp) :: in_q6order_dotmin                                       !q6(i).q6(j) min value to classify as similarily 'bonded' 
    
    integer :: in_sc_flag                                               !signature cells analysis flag for on and off
    real(dp) :: in_sc_cutoff                                            !min. avg. separation for classification accept
    integer :: in_sc_res                                                !angular search resolution 
    integer :: in_sc_cells                                              !number of SC to use
    integer :: in_sc_labs(100000)                                       !list containing SC labels (fix)
    
    integer :: in_SU_flag                                               !structure units analysis flag for on and off
    integer :: in_nnhood(10,4)                                          !selection of environments to track
    
    integer :: in_fradim_atom_centre_flag                               !fractal dimension based on atom centres flag for on and off
    integer :: in_fradim_atom_centre_maxres                             !maximum box division for above    
    integer :: in_fradim_satom_ball_flag                                !fractal dimension based on surface atom ball flag for on and off
    integer :: in_fradim_satom_ball_maxres                              !maximum box division for above
    integer :: in_fradim_np_surf_flag                                   !fractal dimension based on nanoparticle surface flag for on and off
    integer :: in_fradim_np_surf_maxres                                 !maximum box division     
    
    integer :: in_lindem_flag                                           !lindemann index flag for on and off
    integer :: in_lindem_frames                                         !number of frames in lindemann index calculation time average
    
    integer :: analysis_multi                                           !flag to use multi-frame analysis
    integer :: headers_done                                             !flag for writing out initial file headers     

    !Particle type labels 
    character(len=2) :: typelist(con_maxtypes)                          !Label of type i (usually elemental symbols)
    
    !Initial routine parameters   
    character(len=12) :: xyz_prec_str                                   !decimal places of xyz coordinates
    integer :: frame                                                    !frame number for single frame analysis    
    integer :: framecount                                               !counts total frames in movie
    integer :: typecount                                                !counts total particle types in movie 
    
    integer, allocatable :: natoms(:)                                   !particles in ith frame 
    integer, allocatable :: nsurfs(:)                                   !surface particles in the ith frame
    integer, allocatable :: ntypes(:,:)                                 !particles of ith type in jth frame    
    integer :: natoms_max                                               !maximum particles in any frame     
    integer :: nsurfs_max                                               !maximum no. of surface particles in any frame
    integer :: ntypes_max(con_maxtypes)                                 !maximum no. of particles in any frame of type i
    
    integer, allocatable :: flag_surf(:)                                !flag if particle i is a surface particle   
    
    integer :: pbc_option                                               !periodic boundaries flag
    real(dp) :: max_x, min_x, max_y, min_y, max_z, min_z                !particle max and min position across movie
    real(dp) :: mid_x, mid_y, mid_z                                     !particle middle position
    real(dp) :: xl,yl,zl,xl2,yl2,zl2,xl2inv,yl2inv,zl2inv               !cell parameters 
    
    real(dp) :: in_cutoff2(con_maxtypes,con_maxtypes)                   !square of in_cutoff
    
    character(len=2), allocatable :: lc(:)                              !two letter element label of particle i 
    real(dp), allocatable :: xc(:), yc(:), zc(:)                        !x coordinate for frame i, particle j.  Same for y and z
    

    integer, allocatable :: ncord(:)                                    !coordination of particle i using input file cutoffs
    integer, allocatable :: ncordA(:,:,:)                               !coordination of particle i between types j and k using input file cutoffs
    real(dp), allocatable :: gcn(:)                                     !generalised coordination number of particle i using input file cutoffs
    real(dp), allocatable :: gcnA(:,:,:)                                !generalised coordination number of particle i between types j and k using input file cutoffs
    integer, allocatable :: nnlist(:,:)                                 !for particle i, particle label of jth nearest neigbour
    integer, allocatable :: ncordS(:)                                   !surface layer coordination of particle i using input file cutoffs (non-zero if on surface)
    integer, allocatable :: ncordSA(:,:,:)                              !surface layer coordination of particle i between types j and k using input file cutoffs (non-zero if on surface)
    real(dp), allocatable :: gcnS(:)                                    !surface layer generalised coordination number of particle i using input file cutoffs (non-zero if on surface)
    real(dp), allocatable :: gcnSA(:,:,:)                               !surface layer generalised coordination number of particle i between types j and k using input file cutoffs (non-zero if on surface)
    integer, allocatable :: nnlistS(:,:)                                !for surface particle i, particle label of jth nearest surface neighbor
    integer, allocatable :: c_lab(:)                                    !type label of particle           
    

    !Multi-frame variables   
    character(len=2), allocatable :: lca(:,:)                           !two letter element label of particle j in frame i 
    real(dp), allocatable :: xca(:,:), yca(:,:), zca(:,:)               !coordinates of frame i and particle j
    integer, allocatable :: c_laba(:,:)                                 !particle type j in frame i 
    integer, allocatable :: flag_surfa(:,:)                             !multi-frame     



END MODULE VARIABLES


!END OF VARIABLES MODULE
!=======================================================================



!=======================================================================
!START OF NCPAC PROGRAM

PROGRAM NCPAC
    
    use VARIABLES, only: analysis_multi, frame, headers_done,          &  
    in_frames_start, in_frames_end, in_frames_jump,                    &
    in_write_screen, in_fradim_atom_centre_flag,                       &
    in_fradim_satom_ball_flag, in_fradim_np_surf_flag, in_lindem_flag, &
    in_option_cluster, in_option_surf, in_q6order_flag, in_SC_flag,    &
    in_SU_flag, natoms
    IMPLICIT NONE


    !Initial analysis setup from input file    
    call INITIAL
    
    !Single-frame analysis calcs      
    call SINGLE_ALLOCATE      
    call SINGLE_FILE_OPEN

    do frame=1, in_frames_end   
        
        call SINGLE_READIN_XYZ
        
        if(frame>=in_frames_start.and.mod(frame,in_frames_jump)==0) then
            
            call SINGLE_NNLIST
            
            !Filter out clusters below a specified size    
            if(in_option_cluster==1) then
                call CAL_CLUSTER
                call SINGLE_NNLIST
            end if
            
            !Write out labels to featureset file
            call FEATUREFILE
            !Find surface        
            if(in_option_surf==1)  call CAL_SURF
            
            !Write out xyz file with filtering and surface designation for multi-file analysis
            call SINGLE_XYZ_OUT
            !Non-optional analysis
            call CAL_CHAIN
            call CAL_COORD  
            call CAL_BLENGTH
            call CAL_BTYPE
            call CAL_GRSQ
            call CAL_G3
            call CAL_BANGLE
            call CAL_BTORSION
            !call CAL_RINGS   !in development
            
            !Temporary specific projects        
            !call SPC_SLICEXYZ    
            
            !Optional analysis            
            if(in_SU_flag==1)                  call CAL_SU
            if(in_fradim_atom_centre_flag==1)  call CAL_FRACTAL_ATOM_CENTRE
            if(in_fradim_satom_ball_flag==1)   call CAL_FRACTAL_SATOM_BALL
            if(in_fradim_np_surf_flag==1)      call CAL_FRACTAL_NP_SURF
            if(in_q6order_flag==1)             call CAL_Q6 
            if(in_SC_flag==1)                  call CAL_SC 
            
            if(in_write_screen==1) then
                call OUT_PROGRESS                                       !output to screen every 10%
            end if
            
            headers_done = 1                                            !1st write out of file headers done
            
            write(90,*)                                                 !New line for featureset file  
            
        end if  

    end do 
    
    call SINGLE_FILE_CLOSE    
    call SINGLE_DEALLOCATE 
    
    
    !Multi-frame analysis calcs      
    if(analysis_multi==1) then
        call MULTI_ALLOCATE 
        call MULTI_FILE_OPEN
        call MULTI_READIN_XYZ
        
        if(in_lindem_flag==1) call CAL_LINDEMANN
        
        
        call MULTI_DEALLOCATE
        call MULTI_FILE_CLOSE                
    end if 
    

    !Write out to screen       
    if(in_write_screen==1) then
        print*,' '
        print*,'-Completed'
        print*,' '
    end if

    
END PROGRAM NCPAC

!END OF NCPAC PROGRAM
!=======================================================================



!=======================================================================
!START OF INITIAL ROUTINE
!Read input file and set up cell parameters including auto modes

subroutine INITIAL
    
    use VARIABLES  
    IMPLICIT NONE  
    
    
    integer :: i
    
    
    !Zero header write flag
    headers_done = 0
    
    
    !Open input file
    open(10,file='NCPac.inp',status='old')
    
    read(10,*) in_filexyz                                               !xyz filename
    read(10,*) in_xyz_prec                                              !xyz coordinates decimal places
    read(10,*) in_write_screen                                          !write to screen flag
    read(10,*) in_5th_option                                            !read in 5th column surface data
    
    !Determine number of decimal places to be reported
    write(xyz_prec_str,*) in_xyz_prec

    !Write out version
    if(in_write_screen==1) then
        print*,'NCPac - Version: ',con_vers     
    end if     
    
    !Determine no. of frames in movie     
    call INITIAL_FRAMES        
    
    !Determine no. of particles types in movie      
    call INITIAL_TYPES  
    
    !Define bond length cutoffs
    call INITIAL_CUTOFFS        
    
    !Depad xyz movie if padded
    call INITIAL_DEPAD      
    
    !Determine no. of total and surface particles in movie
    call INITIAL_PARTICLES  
    
    !Determine cell lengths
    call INITIAL_CELL    
    
    !Read in no. of frames in analysis 
    read(10,*) in_frames_start, in_frames_end, in_frames_jump          
    if(in_frames_end>framecount) then
        print*,'ERROR - No of frames in analysis > frames in XYZ file'
        STOP
    end if
    
    !reduced density (either a single real across all frames or has to be 
    !calculated later for each frame (usually with varying xl or natoms)
    read(10,*) in_density  
    
    
    !g(r) spacing input
    read(10,*) in_delr, in_gr_points  
    read(10,*) in_sq_points
    
    
    !Read ins for other options       
    read(10,*) in_option_cluster 
    read(10,*) in_mincluster      
    read(10,*) in_option_surf                                                                                
    read(10,*) in_cone_angle                                                                         
    read(10,*) in_surf_points 
    
    read(10,*) in_q6order_flag
    read(10,*) in_q6order_dotmin
    
    read(10,*) in_sc_flag                                        
    read(10,*) in_sc_cutoff                                       
    read(10,*) in_sc_res  
    read(10,*) in_sc_cells
    
    if(in_sc_cells>0) then
        read(10,*) (in_sc_labs(i),i=1,in_sc_cells)              
    end if
    if(in_sc_cells==0.or.in_sc_cells==-1) then
        read(10,*)
    end if  
    
    
    
    
    !Read in nn environments to track  
    read(10,*) in_SU_flag   
    
    !ERROR - Analysis not defined for more than 3 particle types 
    if(in_SU_flag==1.and.typecount>3) then
        print*,' '
        print*,'ERROR: SU routine undefined for more than 3 particle types. Set in_SU_flag=0'
        STOP
    end if 
    
    read(10,*) in_nnhood(1,1),in_nnhood(1,2),in_nnhood(1,3),in_nnhood(1,4)
    read(10,*) in_nnhood(2,1),in_nnhood(2,2),in_nnhood(2,3),in_nnhood(2,4)
    read(10,*) in_nnhood(3,1),in_nnhood(3,2),in_nnhood(3,3),in_nnhood(3,4)
    read(10,*) in_nnhood(4,1),in_nnhood(4,2),in_nnhood(4,3),in_nnhood(4,4)
    read(10,*) in_nnhood(5,1),in_nnhood(5,2),in_nnhood(5,3),in_nnhood(5,4)
    read(10,*) in_nnhood(6,1),in_nnhood(6,2),in_nnhood(6,3),in_nnhood(6,4)
    read(10,*) in_nnhood(7,1),in_nnhood(7,2),in_nnhood(7,3),in_nnhood(7,4)
    read(10,*) in_nnhood(8,1),in_nnhood(8,2),in_nnhood(8,3),in_nnhood(8,4)
    read(10,*) in_nnhood(9,1),in_nnhood(9,2),in_nnhood(9,3),in_nnhood(9,4)
    read(10,*) in_nnhood(10,1),in_nnhood(10,2),in_nnhood(10,3),in_nnhood(10,4)    
    
    read(10,*) in_fradim_atom_centre_flag
    read(10,*) in_fradim_atom_centre_maxres  
    read(10,*) in_fradim_satom_ball_flag
    read(10,*) in_fradim_satom_ball_maxres  
    read(10,*) in_fradim_np_surf_flag
    read(10,*) in_fradim_np_surf_maxres
    
    read(10,*) in_lindem_flag
    read(10,*) in_lindem_frames   
    
    
    
    !Write out options to screen      
    if(in_write_screen==1) then
        if(in_option_cluster==1) then
            print*,' '  
            if(in_mincluster>0) print*,'-Clusters below',in_mincluster,'particles removed'
            if(in_mincluster==-1) print*,'-Clusters removed retaining only largest cluster'
            print*,' '
        end if
        if(in_option_surf==1) then
            print*,' '
            print "(a30,f10.1,a20,i10,a30)",' -Surface particle search using',in_cone_angle,       &
            'degree cone with',in_surf_points,'sphere point distribution'
            print*,' '
        end if        
    end if       
    
    !Single or/and multi frame analysis
    if(in_lindem_flag==1) analysis_multi = 1  
    
    
    
    
    close(10)      
    
    return
end 


!END OF INITIAL ROUTINE
!=======================================================================



!=======================================================================
!START OF INITIAL_FRAMES ROUTINE
!Determines the number of frames in the movie (framecount) 

subroutine INITIAL_FRAMES
    
    use VARIABLES  
    IMPLICIT NONE 
    character (len=500) :: str1,str2
        
    integer :: i, j, k
    
    
    !Determines frames in movie
    open(1, file = in_filexyz, status = 'old')      
    
    framecount = 0
    
    !Remove white space from particle number input 
    100   str2=''
    read(1,'(a500)',END=200) str1
    read(1,*) 
    do k=1,len(str1)   
        if(str1(k:k)/=' ') str2=trim(str2)//trim(str1(k:k))   
    end do
    read(str2,*) j
    
    framecount = framecount + 1
    
    
    do i=1, j
        read(1,*)
    end do
    
    goto 100
    200   continue      
    close(1)
    
    
    !Write out data
    if(in_write_screen==1) then
        print*,' '
        print*,'-Frames in XYZ file:     ',framecount      
    end if
    

    return
end 


!END OF INITIAL_FRAMES ROUTINE
!=======================================================================



!=======================================================================
!START OF INITIAL_TYPES ROUTINE
!Determines the number of particle types (typecount) in movie and 
!stores their types in typelist(i) 

subroutine INITIAL_TYPES
    
    use VARIABLES   
    IMPLICIT NONE 
    
    integer :: i, j, k
    integer :: loop
    integer :: typeadd
    character (len=2) :: temp_l
    character (len=500) :: str1,str2      
    
    
    !Determines types in movie
    open(1, file = in_filexyz, status = 'old')   
    
    typecount = 0 
    do i=1, framecount
        
        !Remove white space from particle number input 
        str2='' 
        read(1,'(a500)') str1
        read(1,*) 
        do k=1,len(str1)   
            if(str1(k:k)/=' ') str2=trim(str2)//trim(str1(k:k))   
        end do
        read(str2,*) loop        
        
        
        do j=1, loop
            
            read(1,*) temp_l
            
            if(typecount==0) then                                     !first type 
                typecount = 1
                typelist(1) = temp_l
            end if
            
            typeadd = 1
            do k=1, typecount
                if(temp_l==typelist(k)) then
                    typeadd = 0      
                end if        
            end do
            if(typeadd==1) then
                typecount = typecount + 1
                if(typecount>100) then
                    print*,' '
                    print*,'ERROR - Particle types exceeded 100'
                    print*,' '
                    STOP
                end if
                typelist(typecount) = temp_l
            end if         
            
        end do
    end do
    
    close(1)
    
    
    
    return
end 


!END OF INITIAL_TYPES ROUTINE
!=======================================================================



!=======================================================================
!START OF INITIAL_CUTOFFS ROUTINE
!Determine bond length cutoffs

subroutine INITIAL_CUTOFFS
    
    use VARIABLES 
    IMPLICIT NONE 
    
    
    integer :: i, j
    integer :: labelfound
    
    character(len=2) :: label_temp(con_maxtypes)                        !temporary labels store for compare 
    
    
    
    !Read in particle label and cutoff matrix       
    do i=1, typecount
        read(10,*) label_temp(i),(in_cutoff(i,j),j=i,typecount)              
    end do
    
    !Check that particle labels in input file are same as those in xyz file
    do i=1, typecount
        labelfound = 0
        do j=1, typecount
            
            if(label_temp(i)==typelist(j)) then
                labelfound = 1
            end if
            
        end do
        if(labelfound==0) then
            print*,'ERROR - Particle Label mismatch in XYZ file and input file' 
            STOP
        end if
    end do
    
    !Relabel typelist(i) to have same order as input file.  Typelist is read on first find basis.
    do i=1, typecount
        typelist(i) = label_temp(i)
    end do
    
    
    !Fill in bond lengths of off diagnals
    do i=1, typecount
        do j=i+1, typecount          
            in_cutoff(j,i) = in_cutoff(i,j)   
        end do
    end do
    
    
    !Square of bond lengths            
    do i=1, typecount
        do j=1, typecount
            in_cutoff2(i,j) = in_cutoff(i,j) * in_cutoff(i,j)
        end do
    end do        
    
    
    !Write out data
    if(in_write_screen==1) then
        print*,' '
        print*,'-No. of particle types:  ',typecount
        do i=1, typecount
            print*,' Type, Label:  ', i, typelist(i)
        end do
    end if     
    
    
    !Write out data
    if(in_write_screen==1) then
        print*,' '
        print*,'-Cutoff Distances (Angstrom)'
        do i=1, typecount
            do j=i, typecount
                
                print "(i10,a3,i10,a3,f12.3)",i,typelist(i), j,typelist(j), in_cutoff(i,j)
                !write(*,1234) typelist(i)
                !1234 format(f5.2)
                
            end do
        end do 
    end if        
    
    
    return
end 

!END OF INITIAL_CUTOFFS ROUTINE
!=======================================================================



!=======================================================================
!START OF INITIAL_DEPAD ROUTINE
!Removes 'padding' particles which are particles located on top of
!each other usually used to keep visualization software like VMD
!happy. Padded particle appear at the end of an xyz frame. 

subroutine INITIAL_DEPAD
    
    use KINDS, only: dp
    use VARIABLES   
    IMPLICIT NONE 
    
    integer :: i, j
    integer :: make_depadded_file                                       !flag to make new file if duplicate found in frame
    integer :: new_n(framecount)                                        !no. particles in depadded frame i
    integer :: temp_n, dupfound
    real(dp) :: temp_x, temp_y, temp_z
    real(dp) :: old_x, old_y, old_z
    character(len=2) :: temp_l  
    character(len=100) :: out_filexyz                                   !filename for depadded xyz
    
    
    !Set make_padded_file to zero (don't make new file)
    make_depadded_file = 0 
    
    
    
    !Open original xyz file 
    open(1, file = in_filexyz, status = 'old') 
    
    new_n = 0
    do i=1, framecount
        
        read(1,*) temp_n
        read(1,*)                     !empty space
        
        dupfound = 0
        do j=1, temp_n
            
            read(1,*) temp_l,temp_x,temp_y,temp_z
            
            if(j>1.and.dupfound==0) then
                if(temp_x==old_x.and.temp_y==old_y.and.temp_z==old_z) then
                    new_n(i) = j-2   !need to remove original and its duplicate found
                    dupfound = 1     !flag that duplicate is found and record location
                end if
            end if
            
            old_x = temp_x
            old_y = temp_y
            old_z = temp_z
            
        end do
        
        !If duplicate not found, new particle count for frame same as old        
        if(dupfound==0) new_n(i) = temp_n
        
        !If duplicate found in frame, flag to make new depadded file        
        if(dupfound==1) make_depadded_file = 1
        
    end do  
    

    
    if(make_depadded_file==1) then 
        
        rewind(1)
        
        out_filexyz = 'xyz_depadded.xyz'
        open(2, file = out_filexyz, status = 'unknown') 
        
        do i=1, framecount
            read(1,*) temp_n
            read(1,*)
            
            write(2,'(i10,/)') new_n(i)
            
            do j=1, temp_n
                
                read(1,*) temp_l,temp_x,temp_y,temp_z    
                if(j<=new_n(i)) then
                    write(2,'(a,2x,3(F8.3,2X))') temp_l,temp_x,temp_y,temp_z  
                end if
                
                
            end do  
            
        end do
        
        !Use the depadded file as the new input file      
        in_filexyz = 'xyz_depadded.xyz'
        
        close(2)
        
    end if
    
    
    close(1) 
    
    
    
    
    
    
    !Write out data
    if(in_write_screen==1) then
        if(make_depadded_file==1) then
            print*,' '
            print*,'-File depadded (duplicates removed in each frame)'
        end if
    end if  
    
    
    
    return
end 


!END OF INITIAL_DEPAD ROUTINE
!=======================================================================



!=======================================================================
!START OF INITIAL_PARTICLES ROUTINE
!For all frames determines the 
!-total number of particles, natoms(frame)
!-total number of surface particles, nsurfs(frame)
!-total number of particles of each particle type, ntypes(type,frame)
!-maximum number of particles for the above in all frames, natoms_max, nsurfs_max, ntypes_max
    
subroutine INITIAL_PARTICLES
    
    use KINDS, only: dp
    use VARIABLES    
    IMPLICIT NONE 
    
    
    integer :: i, j, k, l 
    integer :: temp_flag      
    real(dp) :: temp_x, temp_y, temp_z
    character(len=2) :: temp_l          
    
    
    !Allocate array (natoms)
    ALLOCATE(natoms(framecount))
    ALLOCATE(nsurfs(framecount))
    ALLOCATE(ntypes(typecount,framecount))
    
    !Zero arrays
    natoms = 0
    nsurfs = 0  
    ntypes = 0
    
    
    open(1, file = in_filexyz, status = 'old') 
    
    natoms_max = 0
    nsurfs_max = 0
    ntypes_max = 0
    do i=1, framecount
        
        read(1,*) natoms(i)
        read(1,*)                     !empty space
        
        do j=1, natoms(i)
            
            if(in_5th_option==1) then
                read(1,*) temp_l,temp_x,temp_y,temp_z,temp_flag 
            end if
            if(in_5th_option==0) then
                read(1,*) temp_l,temp_x,temp_y,temp_z
            end if 
            
            
            !Count surface particles
            if(temp_flag==1) then
                nsurfs(i) = nsurfs(i) + 1
            end if 
            
            !Count particles of different types           
            do k=1, typecount
                if(temp_l==typelist(k)) then
                    ntypes(k,i) = ntypes(k,i) + 1
                end if
            end do
            
        end do
        
        !Keep track of maximums  
        if(natoms(i)>natoms_max) natoms_max = natoms(i)      
        if(nsurfs(i)>nsurfs_max) nsurfs_max = nsurfs(i)
        do l=1, typecount
            if(ntypes(l,i)>ntypes_max(l)) ntypes_max(l) = ntypes(l,i)
        end do
        
    end do  
    
    close(1) 
    
    
    !Write out data
    if(in_write_screen==1) then
        print*,' '
        print*,'-Max. particles in any frame:                       ',natoms_max
        print*,' Max. surf particles in any frame:                  ',nsurfs_max  
        do i=1, typecount
            print*,' Max. particles of type',i,'in any frame:   ',ntypes_max(i)
        end do
    end if  
    
    
    return
end 


!END OF INITIAL_PARTICLES ROUTINE
!=======================================================================



!=======================================================================
!START OF INITIAL_CELL ROUTINE
!Determine maximum cell dimensions in x, y and z directions.

subroutine INITIAL_CELL
    
    use KINDS, only: dp    
    use VARIABLES    
    IMPLICIT NONE 
    
    integer :: i, j
    real(dp) :: temp_x, temp_y, temp_z
    real(dp) :: dif_x, dif_y, dif_z
    character(len=2) :: temp_l 
    character(len=80) :: prec_str
    

    
    !Determine maximum and minimum particle x,y,z coordinates
    open(1, file = in_filexyz, status = 'old')
    
    do i=1, framecount
        
        read(1,*) natoms(i)
        read(1,*)                     !empty space
        
        do j=1, natoms(i)
            
            read(1,*) temp_l,temp_x,temp_y,temp_z
            
            !Initialize cell dimensions from 1st particle        
            if(i==1.and.j==1) then
                min_x = temp_x
                min_y = temp_y
                min_z = temp_z
                max_x = temp_x
                max_y = temp_y
                max_z = temp_z
            end if
            
            !Determine min and max cell values
            if(temp_x<min_x) then
                min_x = temp_x
            end if
            if(temp_y<min_y) then
                min_y = temp_y
            end if
            if(temp_z<min_z) then
                min_z = temp_z
            end if
            
            if(temp_x>max_x) then
                max_x = temp_x
            end if
            if(temp_y>max_y) then
                max_y = temp_y
            end if
            if(temp_z>max_z) then
                max_z = temp_z
            end if
            
        end do
    end do
    
    close(1)
    
    !Different between max and min particle positions      
    dif_x = max_x - min_x
    dif_y = max_y - min_y
    dif_z = max_z - min_z
     
    !Centre of particle
    mid_x = dif_x / 2 + min_x
    mid_y = dif_y / 2 + min_y
    mid_z = dif_z / 2 + min_z
    
    !Read in cell lengths      
    read(10,*) in_xl, in_yl, in_zl
    in_xl = dif_x + 5
    in_yl = dif_y + 5
    in_zl = dif_z + 5
    
    !Check to ensure all particle within PBCs
    if(in_xl<=dif_x.or.in_yl<=dif_y.or.in_zl<=dif_z) then
        print*,'ERROR - PBC are too small'
        print*,'in_xl,in_yl,in_zl :',in_xl,in_yl,in_zl
        print*,'dif_x,dif_y,dif_z :',dif_x,dif_y,dif_z
        STOP
    end if
    
    xl = in_xl
    yl = in_yl
    zl = in_zl
    
    xl2 = xl / 2.0_dp
    xl2inv = 1.0_dp / xl2
    
    yl2 = yl / 2.0_dp
    yl2inv = 1.0_dp / yl2
    
    zl2 = zl / 2.0_dp
    zl2inv = 1.0_dp / zl2     
    

    !Write out data
    if(in_write_screen==1) then
        print*,' '
        print*,'-Particle position range'
        prec_str = '(a30,f15.'//xyz_prec_str//',f15.'//xyz_prec_str//',f15.'//xyz_prec_str//',f15.'//xyz_prec_str//')'
        print prec_str,' Xmin, Xmax, Xdif, X_mid:    ',min_x, max_x, dif_x, mid_x
        print prec_str,' Ymin, Ymax, Ydif, Y_mid:    ',min_y, max_y, dif_y, mid_y
        print prec_str,' Zmin, Zmax, Zdif, Z_mid:    ',min_z, max_z, dif_z, mid_z
        print*,' '
        print*,'-Cell dimensions'
        print "(a25,f15.3,f15.3,f15.3)",' xl, yl, zl:          ', xl, yl, zl
    end if  

    return
end 

!END OF INITIAL_CELL ROUTINE
!=======================================================================



!=======================================================================
!START OF SINGLE_ALLOCATE
!Allocate single frame arrays

subroutine SINGLE_ALLOCATE
    
    use VARIABLES  
    IMPLICIT NONE 
    
    !allocate arrays
    ALLOCATE(xc(natoms_max))
    ALLOCATE(yc(natoms_max))
    ALLOCATE(zc(natoms_max))
    ALLOCATE(lc(natoms_max))
    ALLOCATE(c_lab(natoms_max))
    ALLOCATE(flag_surf(natoms_max))
    
    ALLOCATE(ncord(natoms_max))
    ALLOCATE(ncordA(natoms_max,10,10))
    ALLOCATE(gcn(natoms_max))
    ALLOCATE(gcnA(natoms_max,10,10))
    ALLOCATE(nnlist(natoms_max,con_maxnn))
    ALLOCATE(ncordS(natoms_max))
    ALLOCATE(ncordSA(natoms_max,10,10))
    ALLOCATE(gcnS(natoms_max))
    ALLOCATE(gcnSA(natoms_max,10,10))
    ALLOCATE(nnlistS(natoms_max,con_maxnn))
    
    
    return
end 



!END OF SINGLE_ALLOCATE ROUTINE
!=======================================================================



!=======================================================================
!START OF SINGLE_FILE_OPEN ROUTINE

subroutine SINGLE_FILE_OPEN
    
    use VARIABLES        
    IMPLICIT NONE 
  
    !xyz input file  
    open(11,status="old",file=in_filexyz)                               
    
    !Coordination analysis output files      
    open(20,status="unknown",file='od_COORD.csv')
    open(21,status="unknown",file='ov_COORD_total.xyz') 
    open(22,status="unknown",file='ov_COORD_surface.xyz') 
    
    !Bond length and types analysis output files
    open(25,status="unknown",file='od_BOND_types.csv')
    open(26,status="unknown",file='od_BOND_length.csv') 
    open(27,status="unknown",file='ov_BOND_length.xyz')                 !bond length statistics around particle      
    
    !Chain lengths analysis    
    open(28,status="unknown",file='od_CHAIN.csv')        
    open(29,status="unknown",file='ov_CHAIN.xyz')    
    
    !Surface layer output files      
    open(30,status="unknown",file='od_SURF_geometry.csv')               !COM to surface histogram statistics     
    open(31,status="unknown",file='od_SURF_classify.csv')               !surface type and average curvature histograms
    open(32,status="unknown",file='od_SURF_curvature.csv')              !surface particle curvature distribution
    open(33,status="unknown",file='ov_SURF_layer.xyz')                  !xyz of designated surface particles                 
    open(34,status="unknown",file='ov_SURF_classify.xyz')               !xyz output of surface classification 
    open(35,status="unknown",file='ov_SURF_curvature.xyz')              !xyz output of surface local curvature
    
    !1st nn environments population      
    if(in_SU_flag==1) then
        open(40,status="unknown",file='od_SU.csv')           
        open(41,status="unknown",file='od_SU_track.csv')  
        open(42,status="unknown",file='ov_SU_track.xyz')  
    end if
    
    
    if(in_option_cluster==1) then 
        open(45,status="unknown",file='od_CLUSTER_size.csv')            !cluster population histogram
        open(46,status='unknown',file='od_CLUSTER_coord.csv')           !cluster population avg. coordination 
        open(47,status='unknown',file='ov_CLUSTER_no.xyz')              !cluster no
        open(48,status='unknown',file='ov_CLUSTER_filtered.xyz')    !xyz file after filtering  
    end if
    
    !g2(r) file
    open(51,status="unknown",file='od_GR.csv')    
    !g3(theta) file
    open(52,status="unknown",file='od_G3.csv')
    !bond angle (2nd order) file
    open(53,status="unknown",file='od_BOND_angle.csv')
    !bond torsion file
    open(54,status="unknown",file='od_BOND_torsion.csv')
    !S(q) file
    open(55,status="unknown",file='od_SQ.csv')      

    !bond angle and torsion statistics around particle
    open(56,status="unknown",file='ov_BOND_angle1.xyz')
    open(57,status="unknown",file='ov_BOND_angle2.xyz')
    open(58,status="unknown",file='ov_BOND_torsion.xyz')
    open(59,status="unknown",file='ov_RADIAL_distance.xyz')
    
    !q6 analysis output files  
    if(in_q6order_flag==1) then   
        open(60,status="unknown",file='od_Q6Q6.csv')                    !q6.q6 histogram
        open(61,status="unknown",file='ov_Q6Q6.xyz')                    !q6.q6 xyz output    
    end if  
    
    !Signature cells output files
    if(in_sc_flag==1) then
        open(70,status="unknown",file='od_SC_hist.dat')                 !SC classification histogram  
        open(71,status="unknown",file='ov_SC_class.xyz')                !SC classification xyz 
        open(72,status="unknown",file='ov_SC_fits.xyz')
    end if
    
    !Fractal dimension output files
    if(in_fradim_atom_centre_flag==1) then 
        open(80,status="unknown",file='od_FRADIM_ATOM_CENTRE.csv')  
    end if
    if(in_fradim_satom_ball_flag==1) then 
        open(81,status="unknown",file='od_FRADIM_SATOM_BALL.csv')  
    end if
    if(in_fradim_np_surf_flag==1) then 
        open(82,status="unknown",file='od_FRADIM_NP_SURF.csv')  
    end if

    !Universal output file
    open(90,status="unknown",file='od_FEATURESET.csv')      
    
    return
end 


!END OF SINGLE_FILE_OPEN ROUTINE
!=======================================================================



!=======================================================================
!START OF SINGLE_READIN_XYZ ROUTINE

subroutine SINGLE_READIN_XYZ
    
    use VARIABLES    
    IMPLICIT NONE 
    

    !Local
    integer :: i, j
    integer :: flag_correctlabel
    
    
    flag_surf = 0
    
    read(11,*)
    read(11,*)
    do i=1, natoms(frame) 
        
        if(in_5th_option==1) then
            read(11,*) lc(i), xc(i), yc(i), zc(i), flag_surf(i) 
        end if
        
        if(in_5th_option==0) then
            read(11,*) lc(i), xc(i), yc(i), zc(i)
        end if
        
        
        flag_correctlabel = 0
        do j=1, typecount  !loop over components
            if(lc(i)==typelist(j)) then
                flag_correctlabel = 1
                c_lab(i) = j                                            !label particle according to type            
            end if  
        end do
        
        !If input file types different to xyz file label, stop
        if(flag_correctlabel==0) then
            print*,'ERROR - input file label different to xyz label'
            STOP
        end if
        
        
    end do
    
    
    return
end 


!END OF SINGLE_READIN_XYZ ROUTINE
!=======================================================================



!=======================================================================
!START OF SINGLE_NNLIST ROUTINE
!Constructs nearest neightbor list

subroutine SINGLE_NNLIST
    
    use KINDS, only: dp
    use VARIABLES    
    IMPLICIT NONE
    
    
    !Local     
    integer :: i, j, k, l, it1, it2
    real(dp) :: dx, dy, dz ,dr2
    real(dp) :: ncordSum, ncordMax
    real(dp), allocatable :: ncordSumA(:,:), ncordMaxA(:,:)
    
    ALLOCATE(ncordSumA(10,10))
    ALLOCATE(ncordMaxA(10,10))

    ncord = 0
    ncordA = 0
    
    !build neighbor list
    do i=1, natoms(frame)
        do j=(i+1), natoms(frame)
            dx = xc(i) - xc(j)
            dy = yc(i) - yc(j)
            dz = zc(i) - zc(j)
            
            dx = dx -(int(dx*xl2inv)*xl)
            dy = dy -(int(dy*yl2inv)*yl)
            dz = dz -(int(dz*zl2inv)*zl)
            
            dr2 = dx*dx + dy*dy + dz*dz
            
            do it1=1,typecount
                do it2=1,typecount
                    if(c_lab(i)==it1.and.c_lab(j)==it2) then
                        if(dr2<=in_cutoff2(it1,it2)) then
                            
                            !                 -Check for array bound
                            if(ncord(i)>con_maxnn) then
                                print*,'ERROR: con_maxnn too low - too many 1st nn'
                                STOP
                            end if
                            
                            ncord(i) = ncord(i) + 1
                            ncord(j) = ncord(j) + 1
                            ncordA(i,it1,it2) = ncordA(i,it1,it2) + 1
                            ncordA(j,it1,it2) = ncordA(j,it1,it2) + 1
                            nnlist(i,ncord(i)) = j
                            nnlist(j,ncord(j)) = i                    
                        end if
                    end if
                end do
            end do            
            
        end do
    end do 


    !Add off-diagonals because 1-2 == 2-1
    do i=1, natoms(frame)
        do j=1, typecount
            do k=j+1, typecount
                ncordA(i,j,k) = ncordA(i,j,k) + ncordA(i,k,j)
            end do
        end do
    end do


    ! Compute generalised coordination number (take second neighbours into account)
    gcn = 0.0_dp
    gcnA = 0.0_dp
    do i=1, natoms(frame)
        ncordSum = 0.0_dp
        ncordMax = 0.0_dp  !Maximum cn in the bulk
        ncordSumA = 0.0_dp
        ncordMaxA = 0.0_dp
        do j=1, ncord(i)
            it1 = nnlist(i,j)
            ncordSum = ncordSum + ncord(it1)
            if(ncord(it1)>ncordMax) ncordMax = ncord(it1)
            do k=1, typecount
                do l=k, typecount
                    ncordSumA(k,l) = nCordSumA(k,l) + ncordA(it1,k,l)
                    if(ncordA(it1,k,l)>ncordMaxA(k,l)) ncordMaxA(k,l) = ncordA(it1,k,l)
                end do  
            end do  
        end do  

        gcn(i) = ncordSum / ncordMax
        do k=1, typecount
            do l=k, typecount
                gcnA(i,k,l) = ncordSumA(k,l) / ncordMaxA(k,l)
            end do
        end do

    end do

    DEALLOCATE(ncordSumA)
    DEALLOCATE(ncordMaxA)
    
    return
end 


!END OF SINGLE_NNLIST ROUTINE
!=======================================================================



!=======================================================================
!START OF SINGLE_XYZ_OUT ROUTINE
!Write out XYZ file after cluster filtering and with surface particle 
!designation

subroutine SINGLE_XYZ_OUT
    
    use KINDS, only: dp
    use VARIABLES, only: in_option_surf
    use VARIABLES, only: frame, flag_surf, lc, natoms, xc, yc, zc, xyz_prec_str 
    IMPLICIT NONE
    
    !Local    
    integer :: i
    
    !Write out xyz file of filtered clusters
    if(in_option_surf.eq.1) then
        write(48,'(i10,/)') natoms(frame) 
        do i=1,natoms(frame)
            write(48,'(a,2x,3(F10.'//xyz_prec_str//',2X),i10)') lc(i),xc(i),yc(i),zc(i),flag_surf(i)
        end do   
    else    
        write(48,'(i10,/)') natoms(frame) 
        do i=1,natoms(frame)
            write(48,'(a,2x,3(F10.'//xyz_prec_str//',2X))') lc(i),xc(i),yc(i),zc(i)
        end do          
    end if
    
    
    return
end 


!END OF SINGLE_XYZ_OUT ROUTINE
!=======================================================================



!=======================================================================
!START OF SINGLE_FILE_CLOSE ROUTINE

subroutine SINGLE_FILE_CLOSE
    
    use VARIABLES    
    IMPLICIT NONE
    
    
    !XYZ file      
    close(11)
    
    !Coordination analysis output files 
    close(20)
    close(21)
    close(22)
    
    !Bond length and types analysis output files
    close(25)
    close(26)
    close(27)    
    
    !Chain lengths analysis    
    close(28)
    close(29)   
    
    !Surface layer output files      
    close(30)
    close(31)
    close(32)
    close(33)
    close(34)
    close(35)
    
    !1st nn environments population      
    if(in_SU_flag==1) then
        close(40)
        close(41)
        close(42)
    end if  
    
    !Cluster analysis
    if(in_option_cluster==1) then 
        close(45)
        close(46)
        close(47)
        close(48)  
    end if  
    
    
    !Fractal dimension output files
    if(in_fradim_atom_centre_flag==1) then 
        close(80)
    end if
    if(in_fradim_satom_ball_flag==1) then 
        close(81)
    end if
    if(in_fradim_np_surf_flag==1) then 
        close(82)
    end if
    
    !g2(r) file
    close(51)   
    !g3(theta) file
    close(52)
    !g3_2(theta) file
    close(53)     
    !g4(theta) file
    close(54)     
    !S(q) file
    close(55) 

    !Bond angle (1st and 2nd order), torsion and radial distance files
    close(56) 
    close(57) 
    close(58) 
    close(59) 
    
    
    !q6 analysis output files
    if(in_q6order_flag==1) then   
        close(60)
        close(61)
    end if
    
    !Signature cells output files
    if(in_sc_flag==1) then
        close(70)
        close(71)
        close(72)
    end if
    
    close(90)
    
    
    return
end      

!END OF SINGLE_FILE_CLOSE ROUTINE
!=======================================================================



!=======================================================================
!START OF SINGLE_DEALLOCATE

subroutine SINGLE_DEALLOCATE
    
    use VARIABLES    
    IMPLICIT NONE 
    
    
    !Deallocate arrays
    DEALLOCATE(xc)
    DEALLOCATE(yc)
    DEALLOCATE(zc)
    DEALLOCATE(lc)
    DEALLOCATE(c_lab)
    DEALLOCATE(flag_surf)
    
    DEALLOCATE(ncord)
    DEALLOCATE(gcn)
    DEALLOCATE(nnlist)
    
 
    return
end 


!END OF SINGLE_DEALLOCATE ROUTINE
!=======================================================================



!=======================================================================
!START OF MULTI_ALLOCATE

subroutine MULTI_ALLOCATE
    
    use VARIABLES    
    IMPLICIT NONE 
    
    
    ALLOCATE(xca(framecount,natoms_max))
    ALLOCATE(yca(framecount,natoms_max))
    ALLOCATE(zca(framecount,natoms_max)) 
    ALLOCATE(lca(framecount,natoms_max))
    ALLOCATE(c_laba(framecount,natoms_max))   
    ALLOCATE(flag_surfa(framecount,natoms_max))       
    
    
    return
end 



!END OF MULTI_ALLOCATE ROUTINE
!=======================================================================



!=======================================================================
!START OF MULTI_FILE_OPEN ROUTINE

subroutine MULTI_FILE_OPEN
    
    use VARIABLES         
    IMPLICIT NONE 
    

    open(11,status="old",file='ov_CLUSTER_filtered.xyz')

    
    if(in_lindem_flag==1) then
        open(12,status='unknown',file='od_LINDEX.dat')
        open(13,status='unknown',file='ov_LINDEX.xyz') 
    end if  
    
    
    return
end 



!END OF MULTI_FILE_OPEN ROUTINE
!=======================================================================



!=======================================================================
!START OF MULTI_READIN_XYZ ROUTINE

subroutine MULTI_READIN_XYZ
    
    use VARIABLES    
    IMPLICIT NONE 
    
    integer :: i, j, k
    integer :: flag_correctlabel
    
    
    flag_surfa = 0
    
    
    do i=1, framecount
        read(11,*)
        read(11,*)
        
        do j=1, natoms(i)
            
            
            if(in_5th_option==1.or.in_option_surf==1) then
                read(11,*) lca(i,j), xca(i,j), yca(i,j), zca(i,j), flag_surfa(i,j) 
            else
                read(11,*) lca(i,j), xca(i,j), yca(i,j), zca(i,j)
            end if         
           
 
            flag_correctlabel = 0
            do k=1, typecount  !loop over components
                if(lca(i,j)==typelist(k)) then
                    flag_correctlabel = 1
                    c_laba(i,j) = k                                     !label particle according to type            
                end if  
            end do
            
            !If input file types different to xyz file label, stop
            if(flag_correctlabel==0) then
                print*,'ERROR - input file label different to xyz label'
                STOP
            end if          
            
            
            
        end do  
    end do
    
    
    return
end 


!END OF MULTI_READIN_XYZ ROUTINE
!=======================================================================



!=======================================================================
!START OF MULTI_DEALLOCATE ROUTINE

subroutine MULTI_DEALLOCATE
    
    use VARIABLES   
    IMPLICIT NONE 
    
    
    DEALLOCATE(xca)
    DEALLOCATE(yca)
    DEALLOCATE(zca) 
    DEALLOCATE(lca)
    DEALLOCATE(c_laba)
    DEALLOCATE(flag_surfa)
    
    
    return
end 



!END OF MULTI_DEALLOCATE ROUTINE
!=======================================================================



!=======================================================================
!START OF MULTI_FILE_CLOSE ROUTINE

subroutine MULTI_FILE_CLOSE
    
    use VARIABLES    
    IMPLICIT NONE
    
    
    close(11)
    
    if(in_lindem_flag==1) then
        close(12)
        close(13)
    end if  
    
    
    return
end      

!END OF MULTI_FILE_CLOSE ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_CLUSTER ROUTINE

subroutine CAL_CLUSTER
    
    use KINDS, only: dp
    use VARIABLES, only: c_lab, frame, headers_done, in_mincluster,    &
    lc, natoms, natoms_max, ncord, nnlist,xc, yc, zc, xyz_prec_str
    IMPLICIT NONE 
    
    
    !Local     
    integer :: i, j, k
    integer :: mincluster                                               !cluster filter size
    integer :: temp, temp1, temp2, counter, counter2
    integer :: temp_id
    integer :: cluster_maxsize
    integer :: counter_list(natoms_max)
    integer :: counter_list2(natoms_max)
    integer :: cluster_id(natoms_max)
    integer :: cluster_pop(natoms_max)
    integer :: cluster_list(natoms_max)
    integer :: cluster_atoms(natoms_max) 
    
    integer :: cluster_his(natoms_max)
    
    real(dp) :: cluster_his_coord(natoms_max)
    real(dp) :: cluster_coord(natoms_max)
    real(dp) :: temp_xc, temp_yc, temp_zc
    integer :: temp_c_lab
    character(len=2) temp_lc
    
    
    
    
    !Zero arrays etc.
    i=0
    temp_id = 0
    cluster_pop = 0
    cluster_id  = 0
    cluster_list = 0
    cluster_atoms = 0
    cluster_his = 0
    cluster_his_coord = 0.0_dp
    
    
    
    
    temp_id = 0
    do i=1, natoms(frame)
        
        !If particle is not labelled, label it and find all particles in cluster          
        if(cluster_id(i)==0) then
            temp_id = temp_id + 1
            cluster_id(i) = temp_id   
            
            !Initial list from 1st nn of i
            counter = 0
            do j=1, ncord(i)
                temp = nnlist(i,j)
                counter = counter + 1
                counter_list(counter) = temp  
                cluster_id(temp) = temp_id       !label nn             
            end do
            
            !Build out list by adding unlabelled neighors of neighbors         
            100      counter2 = 0
            do j=1, counter           
                temp1 = counter_list(j) 
                
                !if(temp_id==13) print*,'WORKING:',j,temp1,ncord(temp1)
                
                do k=1, ncord(temp1)
                    temp2 = nnlist(temp1,k)  
                    
                    !if not labelled, add to list and label             
                    if(cluster_id(temp2)==0) then
                        counter2 = counter2 + 1
                        counter_list2(counter2) = temp2
                        cluster_id(temp2) = temp_id
                    end if                  
                end do           
            end do
            
            
            !replace loop list with unlabelled list.  Loop until no new particle found
            if(counter2>0) then          
                counter = counter2
                do k=1, counter2
                    counter_list(k) = counter_list2(k)
                end do
                goto 100   
            end if
            
        end if
        
    end do
    
    
    
    !temp_id is the total number of clusters found
    
    
    
    !Cluster size as a function of cluster id and coordination sum of constituting particle            
    do i=1, natoms(frame)
        cluster_pop(cluster_id(i)) = cluster_pop(cluster_id(i)) + 1
        cluster_coord(cluster_id(i)) = cluster_coord(cluster_id(i)) + ncord(i)
    end do
    
    
    
    !Average coordination of cluster i      
    do i=1, temp_id
        cluster_coord(i) = cluster_coord(i) / cluster_pop(i)
    end do
    
    
    !Cluster size histogram (no of clusters versus cluster size)
    do i=1, temp_id
        cluster_his(cluster_pop(i)) = cluster_his(cluster_pop(i)) + 1
        cluster_his_coord(cluster_pop(i)) = cluster_his_coord(cluster_pop(i)) + cluster_coord(i)
    end do
    
    do i=1, natoms(frame)
        if(cluster_his(i)/=0) then
        cluster_his_coord(i) = cluster_his_coord(i) / cluster_his(i)
        else
            cluster_his_coord(i) = 0.0_dp
        end if
    end do
    
    !Maximum cluster size
    cluster_maxsize = 0
    do i=1, natoms(frame)
        if(cluster_his(i)>0) then
            cluster_maxsize = i
        end if
    end do
    
    
    !Filtering choices
    if(in_mincluster==-1) then
        mincluster = cluster_maxsize
    else
        mincluster = in_mincluster 
    end if
    
    
    !Remove clusters smaller than in_mincluster atoms
    counter = 0       
    do i=1, temp_id
        if(cluster_pop(i)>=mincluster.and.mincluster/=0) then            
            counter = counter + 1
            cluster_list(counter) = i  
        end if  
    end do 
    
    
    !Tag atoms that are in cluster bigger than mincluster atoms
    cluster_atoms = 0
    do i=1, natoms(frame)
        do j=1, counter  
            if(cluster_id(i)==cluster_list(j)) then
                cluster_atoms(i) = 1  
            end if
        end do
    end do  
    
    !Write out xyz file of clusters
    write(47,'(i10,/)') natoms(frame) 
    do i=1,natoms(frame)
        write(47,'(a,2x,3(F10.'//xyz_prec_str//',2X),i6)') lc(i),xc(i),yc(i),zc(i),cluster_id(i)
    end do
    
    
    !Write out file titles in first analysis frame  
    if(headers_done==0) then
        write(45,400)    
        400   format('CLUSTER SIZE DISTRIBUTION (frame, no. of clusters)')
        write(45,500)
        500   format('      Frame',$)
        
        do i=1, cluster_maxsize
            write(45,600) i
            600     format(',',i11,$)
        end do
        write(45,*)
    end if
    
    !Write out cluster populations
    write(45,700) frame
    700 format(i11,$)    
    do i=1, cluster_maxsize
        write(45,800) cluster_his(i)
        800   format(',',i11,$)
    end do
    
    write(45,*)
    
    
    !Write out file titles in first analysis frame  
    if(headers_done==0) then
        write(46,900)    
        900   format('CLUSTER AVG. COORDINATION (frame, avg. particle coordination in cluster)')
        write(46,910)
        910   format('      Frame',$)
        
        do i=1, cluster_maxsize
            write(46,920) i
            920     format(',',i11,$)
        end do
        write(46,*)
    end if
    
    !Write out cluster populations
    write(46,930) frame
    930 format(i11,$)    
    do i=1, cluster_maxsize
        write(46,940) cluster_his_coord(i)
        940   format(',',f11.6,$)
    end do
    
    write(46,*)  
    
    
    !Rewrite xyz and labels
    counter = 0
    do i=1, natoms(frame)
        if(cluster_atoms(i)==1) then
            counter = counter + 1
            
            temp_xc = xc(i)
            temp_yc = yc(i)
            temp_zc = zc(i)
            temp_lc = lc(i)
            temp_c_lab = c_lab(i)
            
            xc(counter) = temp_xc
            yc(counter) = temp_yc
            zc(counter) = temp_zc
            lc(counter) = temp_lc
            c_lab(counter) = temp_c_lab
            
        end if
    end do
    
    !put new value of atom count      
    natoms(frame) = counter
    
!    !Write out xyz file of filtered clusters
!    if(mincluster/=0) then
!        write(48,'(i10,/)') natoms(frame) 
!        do i=1,natoms(frame)
!            write(48,'(a,2x,3(F10.'//xyz_prec_str//',2X))') lc(i),xc(i),yc(i),zc(i)
!        end do   
!    end if    
    
    
    
    
    
    return
end 


!END OF CAL_CLUSTER ROUTINE
!=======================================================================



!=======================================================================
!START OF FEATUREFILE ROUTINE

subroutine FEATUREFILE
    
    use KINDS, only: dp
    use VARIABLES, only: con_pi, frame, headers_done, in_delr,         &
    in_gr_points, in_sc_labs, in_option_surf, in_q6order_flag,         &
    in_sc_cells, in_sc_flag, in_sq_points, natoms, typecount  
    IMPLICIT NONE 
    
    
    !Local      
    integer :: i, j, k
    real(dp) delq
    
    
    !Write out header line
    if(headers_done==0) then
    
        !1st line general feature category labels 
        write(90,400) 
        400 format(                                                    &
                                                         1(','),       &!Frame
        'PARTICLES'                                     ,3(','),$)      !N_total
        
        if(in_option_surf==1) then
        write(90,401)
        401 format(                                                    &
        'SURFACE LAYER - Geometry'                      ,7(','),       &!N_bulk, N_surf,!Rmax, Rdif, Ravg, Rdif, Rstd, Rskew, Rkurt
        'SURFACE LAYER - Packing classification'        ,4(','),       &!100, 111, 110, 311
        'SURFACE LAYER - Curvature histogram'           ,180(','),$)    !180 angular points
        end if
        
        write(90,402)
        402 format(                                                    &
        'CHAIN LENGTH HISTOGRAM'                        ,20(','),$)     !20 chain lengths
        
        
        write(90,404)                                                   !Total coords 
        404 format('COORDINATION STATS - All'           ,177(','),$)   
        if(typecount>1) then
            do i=1,typecount                                            !Partial coords
                write(90,405) 
                405 format(                                                & 
                'COORDINATION STATS - Partial'          ,177(','),$)
            end do
        end if
        
        
        write(90,408)                                                   !Total + partials
        408     format(                                                &
        'BOND LENGTH STATS'                             ,7(','),$)        
        do i=1, typecount
            do j=i, typecount
                write(90,410) 
                410     format(                                        &
                'BOND LENGTH STATS'                     ,7(','),$)
            end do
        end do   
         
         

        write(90,415)                                                   !Total bonds              
        415 format(                                                    &
        'BOND TYPE FRACTIONS'                           ,$)
        do i=1, typecount
            do j=i, typecount
            write(90,418)
            418 format(',',$)
            end do
        end do
        write(90,420)                                                   !N bonds column at the end of bond type histogram                           
        420 format(',',$)
        
        
        
        
        write(90,430)                                                   !G2
        430 format('RADIAL DISTRIBUTION FUNCTION G2(r) - Total',$)
        do i=1, in_gr_points+2                                          !+2 for the types columns
            write(90,432)
            432 format(',',$)
        end do
        
        if(typecount>1) then
            do i=1, typecount
                do j=i, typecount
                    write(90,434)
                    434 format('RADIAL DISTRIBUTION FUNCTION G2(r) - Partial',$) 
                    do k=1, in_gr_points+2                                  !+2 for the types column
                        write(90,436)
                        436 format(',',$)
                    end do
                end do
            end do
        end if
        
        
        write(90,440)                                                   !Total structure factor
        440 format('STRUCTURE FACTOR S(q) - Total', $) 
        do i=1, in_sq_points
            write(90,442) 
            442 format(',',$)
        end do 
        
        
        write(90,450)
        450 format('PARTICLE ANGULAR DISTRIBUTION G3(theta) - Total', $) 
        do i=1, 180+2                                                   !+2 for Mean and Std Dev columns
            write(90,452) 
            452 format(',',$)
        end do 
        

        write(90,460)
        460 format('SECOND ORDER PARTICLE ANGULAR DISTRIBUTION G3_2(theta) - Total', $) 
        do i=1, 180+2                                                   !+2 for Mean and Std Dev columns
            write(90,462) 
            462 format(',',$)
        end do 


        write(90,470)
        470 format('PARTICLE TORSION ANGULAR DISTRIBUTION G4(theta) - Total', $) 
        do i=1, 362+4                                                   !2*(0 to 180), +4 for Mean/Std for Pos/Neg angles
            write(90,472) 
            472 format(',',$)
        end do 
        

        if(in_q6order_flag==1) then                                     !q6.q6 coordination      
            write(90,480)
            480 format('Q6.Q6 COORDINATION'             ,69(','),$)
        end if
        
        write(90,490)
        490 format('HERE!',$)
        
        
        

        
        write(90,*)                                                     !new line
        

        
        
        
        !2nd line feature labels
        !Frame number, total particles  
        write(90,500)
        500 format('      Frame',',    N Total',$)
                
        
        !Surface layer analysis 
        if(in_option_surf==1) then        
            
            write(90,510)
            510     format(                                            &
            ',     N Bulk',',     N Surf',',       Rmin',              &
            ',       Rmax',',       Rdif',',       Ravg',              &
            ',       Rstd',',      Rskew',',      Rkurt',              &
            ',        100',',        111',',        110',              &
            ',        311',$) 
            
            !Surface curvature histogram        
            do i=1, 180
                write(90,520) i
                520       format(',',i11,$)
            end do
            
        end if 
        
        
        !Chain length histogram
        do i=1, 20
            write(90,530) i
            530     format(',',i11,$)     
        end do        
        
        
        !Coordination (total first and then any partials)   
        write(90,540)
        540     format(                                            &
        ',       Type',',  Avg Total',',   Avg Bulk',',   Avg Surf',',  Avg Surfo', &
        ', Avg GTotal',',  Avg GBulk',',  Avg GSurf',', Avg GSurfo',                &
        ',         T0',',         T1',',         T2',',         T3',',         T4', &
        ',         T5',',         T6',',         T7',',         T8',',         T9', &
        ',        T10',',        T11',',        T12',',        T13',',        T14', &
        ',        T15',',        T16',',        T17',',        T18',',        T19', &
        ',        T20'                                                              &
        ',         B0',',         B1',',         B2',',         B3',',         B4', &
        ',         B5',',         B6',',         B7',',         B8',',         B9', &
        ',        B10',',        B11',',        B12',',        B13',',        B14', &
        ',        B15',',        B16',',        B17',',        B18',',        B19', &
        ',        B20'                                                              &
        ',         S0',',         S1',',         S2',',         S3',',         S4', &
        ',         S5',',         S6',',         S7',',         S8',',         S9', &
        ',        S10',',        S11',',        S12',',        S13',',        S14', &
        ',        S15',',        S16',',        S17',',        S18',',        S19', &
        ',        S20'                                                              &
        ',        S0o',',        S1o',',        S2o',',        S3o',',        S4o', &
        ',        S5o',',        S6o',',        S7o',',        S8o',',        S9o', &
        ',       S10o',',       S11o',',       S12o',',       S13o',',       S14o', &
        ',       S15o',',       S16o',',       S17o',',       S18o',',       S19o', &
        ',       S20o'                                                              & 
        ',        GT0',',        GT1',',        GT2',',        GT3',',        GT4', &
        ',        GT5',',        GT6',',        GT7',',        GT8',',        GT9', &
        ',       GT10',',       GT11',',       GT12',',       GT13',',       GT14', &
        ',       GT15',',       GT16',',       GT17',',       GT18',',       GT19', & 
        ',       GT20'                                                              &
        ',        GB0',',        GB1',',        GB2',',        GB3',',        GB4', &
        ',        GB5',',        GB6',',        GB7',',        GB8',',        GB9', &
        ',       GB10',',       GB11',',       GB12',',       GB13',',       GB14', &
        ',       GB15',',       GB16',',       GB17',',       GB18',',       GB19', & 
        ',       GB20'                                                              &
        ',        GS0',',        GS1',',        GS2',',        GS3',',        GS4', &
        ',        GS5',',        GS6',',        GS7',',        GS8',',        GS9', &
        ',       GS10',',       GS11',',       GS12',',       GS13',',       GS14', &
        ',       GS15',',       GS16',',       GS17',',       GS18',',       GS19', & 
        ',       GS20'                                                              &
        ',       GS0o',',       GS1o',',       GS2o',',       GS3o',',       GS4o', &
        ',       GS5o',',       GS6o',',       GS7o',',       GS8o',',       GS9o', &
        ',      GS10o',',      GS11o',',      GS12o',',      GS13o',',      GS14o', &
        ',      GS15o',',      GS16o',',      GS17o',',      GS18o',',      GS19o', & 
        ',      GS20o',$)
        
        if(typecount>1) then
            do i=1, typecount
                write(90,540)
            end do
        end if
        
        
        !Bond lengths 
        write(90,545)                                                   !Total particles                                
        545 format(',       Type',',       Type',', Avg Length',            &
                   ',    Std Dev',', Max Length',', Min Length',',  N Lengths',$) 
        do i=1, typecount                                               !Partial particles
            do j=i, typecount
                write(90,550) 
                550     format(',       Type',',       Type',', Avg Length',            &
                               ',    Std Dev',', Max Length',', Min Length',',  N Lengths',$)  
            end do
        end do  
        
        !Bond Types   
        do i=1, typecount
            do j=i, typecount
                write(90,560) i,j 
                560       format(',','      ',i2,' ',i2,$)          
            end do
        end do  
        write(90,570) 
        570   format(',    N Bonds',$)    
        
        
        !g(r) total  
        write(90,580)
        580   format(',       Type',',       Type',$)      
        do i=1, in_gr_points
            write(90,590) (i-1)*in_delr
            590     format(',',f11.4,$)
        end do  
        
        !g(r) partial
        if(typecount>1) then
            do i=1, typecount
                do j=i, typecount   
                    write(90,600)
                    600         format(',       Type',',       Type',$)            
                    do k=1, in_gr_points
                        write(90,610) (k-1)*in_delr
                        610           format(',',f11.4,$)            
                    end do
                end do
            end do   
        end if
        
        
        !S(q) total
        !delq determination from delr
        delq = 2*con_pi / (in_delr*(2*in_sq_points-2)) 
        do i=1, in_sq_points
            write(90,620) (i-1)*delq
            620     format(',',f11.4,$)
        end do
        
        
        !G3 total
        write(90,625) 
        625     format(',        Avg',',    Std Dev',$)
        
        do i=1, 180
            write(90,630) i
            630     format(',',i11,$)
        end do       
        

        !G3_2 total
        write(90,635) 
        635     format(',        Avg',',    Std Dev',$)
        
        do i=1, 180
            write(90,640) i
            640     format(',',i11,$)
        end do       
        
        
        !Bond torsion total
        write(90,655) 
        655     format(',    Neg Avg',',Neg Std Dev',',    Pos Avg',',Pos Std Dev',$)
        
        do i=-180, 0
            write(90,660) i
            660     format(',',i11,$)
        end do       
        
        do i=0, 180
            write(90,670) i
            670     format(',',i11,$)
        end do       



        if(in_q6order_flag==1) then
            write(90,680)
            680     format(                                                             &
            ',  Avg Total',',   Avg Bulk',',   Avg Surf',                               &
            ',         T0',',         T1',',         T2',',         T3',',         T4', &
            ',         T5',',         T6',',         T7',',         T8',',         T9', &
            ',        T10',',        T11',',        T12',',        T13',',        T14', &
            ',        T15',',        T16',',        T17',',        T18',',        T19', &
            ',        T20',',       T>20'                                               &
            ',         B0',',         B1',',         B2',',         B3',',         B4', &
            ',         B5',',         B6',',         B7',',         B8',',         B9', &
            ',        B10',',        B11',',        B12',',        B13',',        B14', &
            ',        B15',',        B16',',        B17',',        B18',',        B19', &
            ',        B20',',       B>20'                                               &
            ',         S0',',         S1',',         S2',',         S3',',         S4', &
            ',         S5',',         S6',',         S7',',         S8',',         S9', &
            ',        S10',',        S11',',        S12',',        S13',',        S14', &
            ',        S15',',        S16',',        S17',',        S18',',        S19', &
            ',        S20',',       S>20',$)             
        end if  
        
        

        !SC  FIX: need to generalize this for search mode
        if(in_sc_flag==1) then
            do i=1, in_sc_cells
                write(90,690) in_sc_labs(i)
                690       format(',',i11,$)                
            end do
        end if
        
        
        
        !New line     
        write(90,*)
        
    end if
    
        
    !Write out frames and total atoms (after filtering)
    write(90,700) frame, natoms(frame) 
    700   format(i11,',',i11,$)
    
    
    
    
    return
end 


!END OF FEATUREFILE ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_SURF ROUTINE

subroutine CAL_SURF
    
    use KINDS, only: dp
    use VARIABLES, only: con_maxnn, in_surf_points, in_5th_option,    &
    natoms_max    
    IMPLICIT NONE
    
    !Local    
    integer :: ncordT(natoms_max)                                       !coordination of particle i using larger cutoffs for surface finding routine
    integer :: nnlistT(natoms_max,con_maxnn)                            !for particle i, particle label of jth nearest neigbour     
    real(dp) :: s_x(in_surf_points)                                     !location of unit sphere points
    real(dp) :: s_y(in_surf_points)
    real(dp) :: s_z(in_surf_points)                                      
    


    !if not reading in 5th column, don't search for surface particles    
    if(in_5th_option==0) then 

        !Generate surface points
        call CAL_SURF_POINTS(s_x, s_y, s_z)

        !Calculate coordination out to to arbitary cutoff distance
        call CAL_SURF_NNLIST(ncordT, nnlistT)  
        
        !Find surface particles      
        call CAL_SURF_FIND(s_x, s_y, s_z, ncordT, nnlistT)
    end if
    
    
    !Surface coordination histogram  
    call CAL_SURF_COORD
    
    !Output surface histogram data
    call CAL_SURF_HISTO       
    
    !Surface curvature and classification
    call CAL_SURF_CURVE
    
    !Output xyz data     
    call CAL_SURF_XYZ  
    
    return
end 

!END OF CAL_SURF ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_SURF_POINTS ROUTINE

subroutine CAL_SURF_POINTS(s_x, s_y, s_z)
    
    use KINDS, only: dp
    use VARIABLES, only: con_pi, in_surf_points
    IMPLICIT NONE 
    
    !Passed    
    real(dp),intent(out) :: s_x(in_surf_points)
    real(dp),intent(out) :: s_y(in_surf_points)
    real(dp),intent(out) :: s_z(in_surf_points)                                  
    
    !Local
    integer :: i
    real(dp) :: c1
    real(dp) :: s_theta(in_surf_points)                    
    real(dp) :: s_phi(in_surf_points)

    
    do i=1, in_surf_points
        c1 = -1.0_dp + 2.0_dp*(i-1.0_dp)/(real(in_surf_points,dp)-1.0_dp)
        s_theta(i) = acos(c1)
        
        if(i==1.or.i==in_surf_points) then
        s_phi(i) = 0.0_dp
        else
            s_phi(i) = ((s_phi(i-1)+3.6_dp/sqrt(in_surf_points*(1.0_dp-c1*c1))))
            s_phi(i) = mod(s_phi(i),2.0_dp*con_pi)
        end if
        
        s_x(i) = cos(s_phi(i))*sin(s_theta(i))
        s_y(i) = sin(s_phi(i))*sin(s_theta(i))
        s_z(i) = cos(s_theta(i))
    end do 
    
    return
end 

!END OF CAL_SURF_POINTS ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_SURF_NNLIST ROUTINE
!Calculates nearest neighbor within twice the cutoff distance

subroutine CAL_SURF_NNLIST(ncordT, nnlistT)

    use KINDS, only: dp    
    use VARIABLES, only: c_lab, con_maxnn, frame, in_cutoff2,          &  
    natoms_max, natoms, typecount,                                      &
    xc, yc, zc, xl, yl, zl, xl2inv, yl2inv, zl2inv 
    IMPLICIT NONE

    !Passed
    integer :: ncordT(natoms_max)                                        
    integer :: nnlistT(natoms_max,con_maxnn)                                 
    
    !Local      
    integer :: i, j, it1, it2
    real(dp) :: dx, dy, dz ,dr2
    real(dp) :: cutoff_scaler
    
    
    ncordT = 0
    
    !Input file cutoff scale multiplier       
    cutoff_scaler = 2.0_dp
    
    !build neighbor list
    do i = 1, natoms(frame)
        do j = (i+1), natoms(frame)
            dx = xc(i) - xc(j)
            dy = yc(i) - yc(j)
            dz = zc(i) - zc(j)
            
            dx = dx -(int(dx*xl2inv)*xl)
            dy = dy -(int(dy*yl2inv)*yl)
            dz = dz -(int(dz*zl2inv)*zl)
            
            dr2 = dx*dx + dy*dy + dz*dz
            
            do it1=1,typecount
                do it2=1,typecount
                    if(c_lab(i)==it1.and.c_lab(j)==it2) then
                        if(dr2<=(in_cutoff2(it1,it2)*cutoff_scaler)) then
                            
                            !Check for array bound
                            if(ncordT(i)>con_maxnn) then
                                print*,'ERROR: con_maxnn too low - too many 1st nn'
                                STOP
                            end if
                            
                            ncordT(i) = ncordT(i) + 1
                            ncordT(j) = ncordT(j) + 1
                            nnlistT(i,ncordT(i)) = j
                            nnlistT(j,ncordT(j)) = i
                        end if
                    end if
                end do
            end do            
            
            
            
        end do
    end do 

    return
end 

!END OF CAL_SURF_NNLIST ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_SURF_FIND ROUTINE
!Find surface atoms using cone method.  If surface atom found
!flag_surf(i) = 1..else its 0.

subroutine CAL_SURF_FIND(s_x, s_y, s_z ,ncordT, nnlistT)
    
    use KINDS, only: dp
    use VARIABLES, only: con_maxnn, frame, flag_surf, in_cone_angle,   &
    in_surf_points, natoms, natoms_max, nsurfs,                        &
    xc, yc, zc, xl, yl, zl, xl2inv, yl2inv, zl2inv
    IMPLICIT NONE
    
    
    
    !Passed
    real(dp),intent(in) :: s_x(in_surf_points)
    real(dp),intent(in) :: s_y(in_surf_points)
    real(dp),intent(in) :: s_z(in_surf_points)                                          
    integer,intent(in) :: ncordT(natoms_max)                                        
    integer,intent(in) :: nnlistT(natoms_max,con_maxnn)     
    
    !Local      
    integer :: i, j, k, it
    real(dp) :: x1, y1, z1, x2, y2, z2, x3, y3, z3
    real(dp) :: dx_12, dy_12, dz_12, dx_13, dy_13, dz_13
    real(dp) :: dr_13, dot_1213, cosangle_1213, angle_1213
    real(dp) :: minangle, maxangle      
    
    
    nsurfs(frame) = 0     
    do i=1, natoms(frame)
        
        x1 = xc(i)
        y1 = yc(i)
        z1 = zc(i)
        
        maxangle = 0.0_dp
        do j=1, in_surf_points
            
            x2 = s_x(j) 
            y2 = s_y(j) 
            z2 = s_z(j) 
            
            
            minangle = 360.0_dp                                         !set initial minimum angle to 360 degrees
            do k=1, ncordT(i)
                it = nnlistT(i,k)
                
                x3 = xc(it)
                y3 = yc(it)
                z3 = zc(it)
                
                
                dx_12 = x2                                              !points assumed to be around particle i
                dy_12 = y2
                dz_12 = z2
                
                dx_13 = x3-x1
                dy_13 = y3-y1
                dz_13 = z3-z1
                
                !PBCs            
                dx_13 = dx_13-(int(dx_13*xl2inv)*xl)
                dy_13 = dy_13-(int(dy_13*yl2inv)*yl)
                dz_13 = dz_13-(int(dz_13*zl2inv)*zl)            
                
                
                !Magnitude of vector 1-2 (i to point) is 1 due to unit sphere
                
                
                !Magnitude of vector 1-3 (i to nn particle)
                dr_13 = sqrt((dx_13*dx_13) + (dy_13*dy_13) + (dz_13*dz_13))
                
                !Angle between 1-2 and 1-3 (cos(theta) = (12).(13)/|12||13|  
                dot_1213 = (dx_12*dx_13+dy_12*dy_13+dz_12*dz_13)
                cosangle_1213 = dot_1213 / dr_13                        !dr_12 = 1 from unit sphere
                angle_1213 = acos(cosangle_1213)*57.29577951_dp
                
                if(angle_1213<minangle) minangle = angle_1213           !min angle for point
                
            end do
            
            
            !Maximum of the minimum angles for points
            if(minangle>maxangle) maxangle = minangle          
            
        end do
        
        
        
        !If max angle found is greater than cutoff, label surface atom         
        if(maxangle>in_cone_angle) then
            flag_surf(i) = 1                                            !flag particle as surface
            nsurfs(frame) = nsurfs(frame) + 1
        end if
        
    end do 
    

    return
end 

!END OF CAL_SURF_FIND ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_SURF_COORD ROUTINE
!Determine Coordination Histogram of surface layer

subroutine CAL_SURF_COORD
    
    use KINDS, only: dp
    use VARIABLES, only: c_lab, con_maxnn, flag_surf, frame,           &
    in_cutoff2, natoms, ncordS, ncordSA, gcnS, gcnSA, nnlistS,         &
    typecount, xc, yc, zc, xl, yl, zl, xl2inv, yl2inv, zl2inv                               
    IMPLICIT NONE
    

    !Local      
    integer :: i, j, k, l, it1, it2
    real(dp) :: dr2, dx, dy, dz
    real(dp) :: ncordSSum, ncordSMax
    real(dp), allocatable :: ncordSSumA(:,:), ncordSMaxA(:,:)
    
    ALLOCATE(ncordSSumA(10,10))
    ALLOCATE(ncordSMaxA(10,10))
    
    
    ncordS = 0
    do i=1, natoms(frame)
        do j=(i+1), natoms(frame)
            
            if(flag_surf(i)==1.and.flag_surf(j)==1) then
                
                dx = xc(i) - xc(j)
                dy = yc(i) - yc(j)
                dz = zc(i) - zc(j)
                
                dx = dx -(int(dx*xl2inv)*xl)
                dy = dy -(int(dy*yl2inv)*yl)
                dz = dz -(int(dz*zl2inv)*zl)
                
                dr2 = dx*dx + dy*dy + dz*dz
                
                do it1=1,typecount
                    do it2=1,typecount
                        if(c_lab(i)==it1.and.c_lab(j)==it2) then
                            if(dr2<=in_cutoff2(it1,it2)) then
                                
                                !                  -Check for array bound
                                if(ncordS(i)>con_maxnn) then
                                    print*,'ERROR: con_maxnn too low - too many 1st nn'
                                    STOP
                                end if
                                
                                ncordS(i) = ncordS(i) + 1
                                ncordS(j) = ncordS(j) + 1
                                ncordSA(i,it1,it2) = ncordSA(i,it1,it2) + 1
                                ncordSA(j,it1,it2) = ncordSA(j,it1,it2) + 1
                                nnlistS(i,ncordS(i)) = j
                                nnlistS(j,ncordS(j)) = i
                            end if
                        end if
                    end do
                end do 
                
            end if
            
        end do
    end do  
    

    !Add off-diagonals because 1-2 == 2-1
    do i=1, natoms(frame)
        do j=1, typecount
            do k=j+1, typecount
                ncordSA(i,j,k) = ncordSA(i,j,k) + ncordSA(i,k,j)
            end do
        end do
    end do

    
    ! Compute generalised coordination number (take second neighbours into account)
    gcnS = 0.0_dp
    gcnSA = 0.0_dp
    do i=1, natoms(frame)
        ncordSSum = 0.0_dp
        ncordSMax = 0.0_dp
        ncordSSumA = 0.0_dp
        ncordSMaxA = 0.0_dp
        do j=1, ncordS(i)
            it1 = nnlistS(i,j)
            ncordSSum = ncordSSum + ncordS(it1)
            if(ncordS(it1)>ncordSMax) ncordSMax = ncordS(it1)
            do k=1, typecount
                do l=k, typecount
                    ncordSSumA(k,l) = nCordSSumA(k,l) + ncordSA(it1,k,l)
                    if(ncordSA(it1,k,l)>ncordSMaxA(k,l)) ncordSMaxA(k,l) = ncordSA(it1,k,l)
                end do  
            end do  
        end do  
        if(ncordSSum<1) cycle
        gcnS(i) = ncordSSum / ncordSMax
    end do
     
    DEALLOCATE(ncordSSumA)
    DEALLOCATE(ncordSMaxA)
    
    return
end 


!END OF CAL_SURF_COORD ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_SURF_HISTO ROUTINE
!Output Surface-Center of Mass Histogram 

subroutine CAL_SURF_HISTO
    
    use KINDS, only: dp
    use VARIABLES, only: frame, flag_surf, headers_done, nsurfs,       & 
    natoms, xc, yc, zc
    IMPLICIT NONE
    
    
    !Local
    integer :: i, bin
    integer :: bins(100000)
    real(dp) binspace

    real(dp) :: SC
    real(dp) :: dis_min, dis_max, dis_range    
    real(dp) :: disx, disy, disz, disr
    real(dp) :: dist, dis_p2, dis_p3, dis_p4
    real(dp) :: dis_MEAN, dis_DEVI, dis_SKEW, dis_KURT
    real(dp) :: xsum, ysum, zsum
    real(dp) :: xcom, ycom, zcom
    
 
    
    !Write out file titles in first analysis frame   
    if(headers_done==0) then
        write(30,100)  
        100   format('SURFACE LAYER - Geometry')
        write(30,200)
        200   format('      Frame',$)
        write(30,300)
        300   format(',       Rmin',',       Rmax',',       Rdif',     &
                     ',       Ravg',',       Rstd',',      Rskew',     &
                     ',      Rkurt',',    N Total',',     N Surf')
    end if
    
    
    
    !define histogram spacing
    binspace = 0.2_dp
    
    !zero histogram array
    bins = 0
        
    !Calculate Center of Mass
    xsum = 0.0_dp
    ysum = 0.0_dp
    zsum = 0.0_dp
    
    !Centre of mass (assumes structures are not passing via periodic boundaries)      
    do i=1, natoms(frame)
        xsum = xsum + xc(i)
        ysum = ysum + yc(i)
        zsum = zsum + zc(i)
    end do
    xcom = xsum / natoms(frame)
    ycom = ysum / natoms(frame)
    zcom = zsum / natoms(frame)
    
    
    
    
    !Max, Min, Mean and create COM to surface histogram
    dist = 0.0_dp
    dis_p2 = 0.0_dp
    dis_p3 = 0.0_dp
    dis_p4 = 0.0_dp
    SC = 0.0_dp
    
    do i=1, natoms(frame)
        if(flag_surf(i)==1) then
            SC = SC + 1.0_dp                                            !Number of surface particles
            
            disx = xcom - xc(i)
            disy = ycom - yc(i)
            disz = zcom - zc(i)
            
            disr = sqrt(disx*disx+disy*disy+disz*disz)
            
            !Maximum COM to surface distance
            if(SC==1.0) dis_max = disr
            if(disr>dis_max) then
                dis_max = disr
            end if
            
            !Minimum COM to surface distance
            if(SC==1.0) dis_min = disr
            if(disr<dis_min) then
                dis_min = disr
            end if
            
            dist = dist + disr
            
            bin = disr / binspace + 1.5
            if(bin>100000) then
              print*,'Error: Bin spacing too small. Increase binspace'
              STOP
            end if
            bins(bin) = bins(bin) + 1
        end if
    end do
    
    dis_range = dis_max - dis_min
    dis_MEAN = dist / SC
    
    
    !	  -Standard Deviation of histogram       
    do i=1, natoms(frame)
        if(flag_surf(i)==1) then
            disx = xcom - xc(i)
            disy = ycom - yc(i)
            disz = zcom - zc(i)
            
            disr = sqrt(disx*disx+disy*disy+disz*disz)
            
            dis_p2 = dis_p2 + (disr-dis_MEAN)**2
        end if
    end do
    dis_DEVI = sqrt(dis_p2 / (SC-1))
    
    !	  -Skewness and Kurtosis of histogram
    do i=1, natoms(frame)
        if(flag_surf(i)==1) then
            disx = xcom - xc(i)
            disy = ycom - yc(i)
            disz = zcom - zc(i)
            
            disr = sqrt(disx*disx+disy*disy+disz*disz)
            
            dis_p3 = dis_p3 + ((disr - dis_MEAN)/dis_DEVI)**3
            dis_p4 = dis_p4 + ((disr - dis_MEAN)/dis_DEVI)**4
        end if
    end do
    dis_SKEW = (SC/((SC-1)*(SC-2)))*dis_p3
    dis_KURT = (SC*(SC+1))/((SC-1)*(SC-2)*(SC-3))*dis_p4
    dis_KURT = dis_KURT - ((3*(SC-1)*(SC-1))/((SC-2)*(SC-3)))
    
    !Write out surface layer histogram stats
    write(30,700) frame,dis_min,dis_max,dis_range,dis_MEAN,dis_DEVI    &
    ,dis_SKEW,dis_KURT,natoms(frame),nsurfs(frame)
    700 format(i11,',',f11.6,',',f11.6,',',f11.6,',',f11.6,            &
                   ',',f11.6,',',f11.6,',',f11.6,',',i11,',',i11)
    
    !Write out to featureset file
    write(90,710) natoms(frame)-nsurfs(frame),nsurfs(frame),dis_min,   &
    dis_max, dis_range,dis_MEAN,dis_DEVI,dis_SKEW,dis_KURT
    710 format(',',i11,',',i11,',',f11.6,',',f11.6,',',f11.6,',',f11.6,&
               ',',f11.6,',',f11.6,',',f11.6,$)      
    
    
    return
end 


!END OF CAL_SURF_HISTO ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_SURF_CURVE ROUTINE

subroutine CAL_SURF_CURVE
    
    use KINDS, only: dp
    use VARIABLES, only: con_maxnn, con_pi, flag_surf, frame,          & 
    headers_done, lc, natoms_max, natoms,                              & 
    ncord, ncordS, nnlist, nnlistS, xc, yc, zc, xyz_prec_str
    IMPLICIT NONE
     
    
    !Local      
    integer :: i, j, k, l
    integer :: temp, temp1
    integer :: flag_reject
    
    integer :: Aindex, Bindex
    integer :: rank_angle(1000), done_angle(1000)
    integer :: small_index, temp_label, temp_list(1000)
    integer, allocatable :: nnclass(:)
    real(dp), allocatable :: nnangle(:,:)
    real(dp), allocatable :: nncurve(:,:)
    real(dp), allocatable :: nncurve_avg(:)
    integer :: nnclass_hist(4), nncurve_hist(180)
    
    real(dp) :: crossx_temp(con_maxnn), crossy_temp(con_maxnn), crossz_temp(con_maxnn)
    integer :: cross_flag(con_maxnn)
    real(dp) :: crossx_sum, crossy_sum, crossz_sum, cross_count
    real(dp) :: x_O, y_O, z_O
    real(dp) :: x_A, y_A, z_A
    real(dp) :: x_B, y_B, z_B
    real(dp) :: dx_AO, dy_AO, dz_AO, dr_AO
    real(dp) :: dx_BO, dy_BO, dz_BO, dr_BO
    real(dp) :: dot_AOBO, angle_AOBO
    real(dp) :: small_angle
    real(dp) :: v1x, v1y, v1z, mag1
    real(dp) :: v2x, v2y, v2z, mag2
    real(dp) :: crossx, crossy, crossz, mag4, mag5
    real(dp) :: dot1, dot2
    real(dp), allocatable :: xbulk(:), ybulk(:), zbulk(:)               !bulk unit vector 
    integer, allocatable :: labatom(:)
    integer :: phis
    
    integer :: angle60count, angle90count
    
    
    
    ALLOCATE(xbulk(natoms_max))
    ALLOCATE(ybulk(natoms_max))
    ALLOCATE(zbulk(natoms_max))
    ALLOCATE(labatom(natoms_max))
    ALLOCATE(nnclass(natoms_max))
    ALLOCATE(nnangle(natoms_max,con_maxnn))
    ALLOCATE(nncurve(natoms_max,con_maxnn)) 
    ALLOCATE(nncurve_avg(natoms_max))
    
    
    !Zero arrays 
    nnclass = 0
    nnangle = 0.0_dp
    nncurve = 0.0_dp
    nncurve_avg = -1.0_dp                                               !1 label for all atoms



    
    !1. Determine volume vector
    !--------------------------
    !Volume unit vector determines the average nn vector from each
    !atom (summing over ALL nn of an atom).  For surface atoms,
    !this points into the bulk and is used later to determine
    !the sign of a cross product.
    
    do i=1, natoms(frame)
        
        if(flag_surf(i)==1) then
            
            
            xbulk(i) = 0.0_dp
            ybulk(i) = 0.0_dp
            zbulk(i) = 0.0_dp
            
            do j=1, ncord(i)
                k = nnlist(i,j)
                xbulk(i) = xbulk(i) + (xc(k)-xc(i))
                ybulk(i) = ybulk(i) + (yc(k)-yc(i))
                zbulk(i) = zbulk(i) + (zc(k)-zc(i))
            end do
            
            mag1 = sqrt(xbulk(i)**2+ybulk(i)**2+zbulk(i)**2)
            xbulk(i) = xbulk(i) / mag1
            ybulk(i) = ybulk(i) / mag1
            zbulk(i) = zbulk(i) / mag1
            
        end if
    end do   
    
    
    
    
    
    !2. Determine ordered surface nn angles
    !-------------------------------------
    do i=1, natoms(frame)
        
        !location of origin (central atom)        
        x_O = xc(i)
        y_O = yc(i)
        z_O = zc(i)
        
        !Zero flag for counted neighbor
        done_angle = 0          
        
        !oop over neighbors  
        
        if(ncordS(i)>1) then   !angles only exist for particle with more than 1 nn          
            
            do j=1, ncordS(i) 
                
                if(j==1) then
                    Aindex = nnlistS(i,1)
                    rank_angle(1) = 1     !1st angle is always 1st in list
                    done_angle(1) = 1     !1st angle is flagged as counted
                end if
                
                if(j>1) then
                    Aindex = nnlistS(i,rank_angle(j))  !index is of last closest atom
                end if
                
                !coordinates of neighbor A            
                x_A = xc(Aindex)
                y_A = yc(Aindex)
                z_A = zc(Aindex)
                
                !find the smallest angle between i-j and i-k            
                small_angle = 2.0_dp * con_pi  !set angle to 2pi
                do k=1, ncordS(i)
                    
                    Bindex = nnlistS(i,k)            
                    
                    if(Aindex/=Bindex) then
                        
                        !coordinates of neighbor B          
                        x_B = xc(Bindex)
                        y_B = yc(Bindex)
                        z_B = zc(Bindex)
                        
                        dx_AO = x_A - x_O
                        dy_AO = y_A - y_O
                        dz_AO = z_A - z_O
                        
                        dx_BO = x_B - x_O
                        dy_BO = y_B - y_O
                        dz_BO = z_B - z_O
                        
                        dr_AO = sqrt(dx_AO*dx_AO+dy_AO*dy_AO+dz_AO*dz_AO)
                        dr_BO = sqrt(dx_BO*dx_BO+dy_BO*dy_BO+dz_BO*dz_BO)
                        
                        dot_AOBO = dx_AO*dx_BO + dy_AO*dy_BO + dz_AO*dz_BO
                        
                        angle_AOBO = acos(dot_AOBO/(dr_AO*dr_BO))
                        
                        if(j==ncordS(i)) done_angle(1) = 0              !last atom shares angle to first, allows loop over last atom               
                        
                        if(done_angle(k)/=1.and.angle_AOBO<small_angle) then
                            small_angle = angle_AOBO
                            small_index = k
                        end if
                        
                    end if  
                end do
                
                done_angle(small_index) = 1      
                rank_angle(j+1) = small_index          
            end do 
            
            !Reorder ncordS (ends up in either clock or anticlock wise direction)
            do l=1, ncordS(i)
                temp_label = rank_angle(l)
                temp_list(l) = nnlistS(i,temp_label)       
            end do
            do l=1, ncordS(i)
                nnlistS(i,l) = temp_list(l)
            end do        
        end if
        
    end do  
    
    
    
    
    
    !3. Determine 1st nn angles
    !-------------------------       
    do i=1, natoms(frame)
        
        x_O = xc(i)
        y_O = yc(i)
        z_O = zc(i)                
        
        
        if(ncordS(i)>1) then
            
            do j=1, ncordS(i)        
                Aindex = nnlistS(i,j)
                
                k = j+1
                if(k>ncordS(i)) k = 1                                   !last atom is first
                Bindex = nnlistS(i,k)
                
                x_A = xc(Aindex)
                y_A = yc(Aindex)
                z_A = zc(Aindex)
                
                x_B = xc(Bindex)
                y_B = yc(Bindex)
                z_B = zc(Bindex)
                
                dx_AO = x_A - x_O
                dy_AO = y_A - y_O
                dz_AO = z_A - z_O
                
                dx_BO = x_B - x_O
                dy_BO = y_B - y_O
                dz_BO = z_B - z_O
                
                dr_AO = sqrt(dx_AO*dx_AO+dy_AO*dy_AO+dz_AO*dz_AO)
                dr_BO = sqrt(dx_BO*dx_BO+dy_BO*dy_BO+dz_BO*dz_BO)
                
                dot_AOBO = dx_AO*dx_BO + dy_AO*dy_BO + dz_AO*dz_BO
                
                angle_AOBO = acos(dot_AOBO/(dr_AO*dr_BO))*(180.0_dp/con_pi)                    
                nnangle(i,j) = angle_AOBO
                
            end do
        end if
    end do
    
    
    
    
    
    !4. Determine local curvature
    !---------------------------
    do i=1, natoms(frame)
        
        crossx_sum = 0.0_dp                                             !zero sum for nn vector crross product
        crossy_sum = 0.0_dp
        crossz_sum = 0.0_dp   
        cross_flag = 0
        cross_count = 0.0_dp
        
        if(ncordS(i)>2) then                                            !define curvature only for particle with nn>2
            
            do j=1, ncordS(i)
                temp = nnlistS(i,j)
                
                k = j+1   
                if(k>ncordS(i)) k = 1                                   !last atom is first          
                temp1 = nnlistS(i,k)
                
                !Unit vector 1 (Rij)
                v1x = xc(temp)-xc(i)
                v1y = yc(temp)-yc(i)
                v1z = zc(temp)-zc(i)
                
                mag1 = sqrt(v1x**2.0_dp+v1y**2.0_dp+v1z**2.0_dp)
                
                v1x = v1x / mag1
                v1y = v1y / mag1
                v1z = v1z / mag1
                
                !Unit vector 2 (Rik)
                v2x = xc(temp1)-xc(i)
                v2y = yc(temp1)-yc(i)
                v2z = zc(temp1)-zc(i)
                
                mag2 = sqrt(v2x**2.0_dp+v2y**2.0_dp+v2z**2.0_dp)
                
                v2x = v2x / mag2
                v2y = v2y / mag2
                v2z = v2z / mag2
                
                
                !Unit cross product of unit vectors Rij and Rik
                crossx = v1y*v2z-v1z*v2y
                crossy = -(v1x*v2z-v1z*v2x)
                crossz = v1x*v2y-v1y*v2x
                
                mag4 = sqrt(crossx**2.0_dp+crossy**2.0_dp+crossz**2.0_dp)
                
                
                
                !if cross product mag is zero, flag it as undefined (example. 100 type surfaces)
                if(mag4>0.0_dp) then
                    cross_flag(j) = 1
                end if
                
                if(mag4>0.0_dp) then
                    crossx = crossx / mag4
                    crossy = crossy / mag4
                    crossz = crossz / mag4
                end if
                
                dot1 = (crossx*xbulk(i)+crossy*ybulk(i)+crossz*zbulk(i))
                
                if(dot1>0.0_dp) then
                    crossx = -crossx
                    crossy = -crossy
                    crossz = -crossz
                end if
                
                !-temporary store nn cross product unit vectors
                crossx_temp(j) = crossx
                crossy_temp(j) = crossy
                crossz_temp(j) = crossz
                
                !-Vector sum of nn cross products (used for average)
                if(cross_flag(j)==1) then
                    crossx_sum = crossx_sum + crossx
                    crossy_sum = crossy_sum + crossy
                    crossz_sum = crossz_sum + crossz
                    cross_count = cross_count + 1.0_dp
                end if
                
            end do
            
            !-average vector of defined cross products
            if(cross_count/=0.0_dp) then
                crossx_sum = crossx_sum / cross_count
                crossy_sum = crossy_sum / cross_count
                crossz_sum = crossz_sum / cross_count
            end if 
            
            !-normalized unit vector of average vector
            mag5 = sqrt(crossx_sum**2.0_dp+crossy_sum**2.0_dp+crossz_sum**2.0_dp)                 
            crossx_sum = crossx_sum / mag5
            crossy_sum = crossy_sum / mag5
            crossz_sum = crossz_sum / mag5 
            
            
            !-dot product of cross product unit vector with cross product vector average
            !used to work out angle
            do j=1, ncordS(i)
                
                dot1 = (crossx_sum*crossx_temp(j)+crossy_sum*crossy_temp(j)+crossz_sum*crossz_temp(j))
                if(dot1<1.000001_dp.and.dot1>0.999999_dp) dot1 = 1.0_dp    !rounding errors
                dot2 = acos(dot1)*57.29577951_dp
                nncurve(i,j) = dot2
                
                !-ignore any 180 degree angles in cross product that led to a zero cross product (set curvature to zero)          
                if(crossx_temp(j)==0.0_dp.and.crossy_temp(j)==0.0_dp.and.crossz_temp(j)==0.0_dp) nncurve(i,j) = 0.0_dp
                
            end do
            
        end if  
        
    end do
    
    
    

    !5. Determine surface type classification
    !----------------------------------------
    do i=1, natoms(frame)
        if(flag_surf(i)==1) then
            
            !-100 packing (coordination=4, minimal curvature, 90 degree angles)
            if(ncordS(i)==4) then                  
                flag_reject = 0 
                do j=1,ncordS(i)             
                    if(nncurve(i,j)>15.0_dp) flag_reject = 1     !FIX
                    if(nnangle(i,j)<70.0_dp.or.nnangle(i,j)>110.0_dp) flag_reject = 1
                end do
                if(flag_reject==0) nnclass(i)=1
            end if
            
            !-111 packing (coordination=6, minimal curvature, 60 degree angles)
            if(ncordS(i)==6) then                  
                flag_reject = 0 
                do j=1,ncordS(i)             
                    if(nncurve(i,j)>15.0_dp) flag_reject = 1  
                    if(nnangle(i,j)<40.0_dp.or.nnangle(i,j)>80.0_dp) flag_reject = 1
                end do
                if(flag_reject==0) nnclass(i)=2
            end if          
            
            !-110 packing (coordination=6, moderate curvature, 60 degree angles)
            if(ncordS(i)==6) then                  
                flag_reject = 0 
                do j=1,ncordS(i)             
                    if(nncurve(i,j)<15.0_dp) flag_reject = 1  
                    if(nnangle(i,j)<40.0_dp.or.nnangle(i,j)>80.0_dp) flag_reject = 1
                end do
                if(flag_reject==0) nnclass(i)=3
            end if   
            
            !-311 packing (coordination = 5, moderate curvature, 2X90deg+3X60deg angles)          
            if(ncordS(i)==5) then          
                flag_reject = 0
                angle60count = 0
                angle90count = 0
                do j=1,ncordS(i)             
                    if(nncurve(i,j)<15.0_dp) flag_reject = 1  
                    if(nnangle(i,j)>50.0_dp.and.nnangle(i,j)<70.0_dp)  angle60count = angle60count + 1
                    if(nnangle(i,j)>80.0_dp.and.nnangle(i,j)<100.0_dp) angle90count = angle90count + 1
                end do
                if(angle60count/=3.or.angle90count/=2) flag_reject = 1            
                if(flag_reject==0) nnclass(i)=4
                
            end if          
        end if
        
    end do
    
    
    
    
    !6. Calculate average particle curvature
    !---------------------------------------
    do i=1, natoms(frame)
        if(flag_surf(i)==1) then
            nncurve_avg(i) = 0.0_dp                                     !0 label for surface atoms
            do j=1, ncordS(i)
                nncurve_avg(i) = nncurve_avg(i) + nncurve(i,j)          !curvature sum
            end do  
            if(ncordS(i)/=0) then
                nncurve_avg(i) = nncurve_avg(i)/ncordS(i)
            end if
        end if
    end do
    
    
    
    
    !7. Calculate surface classification and average curvature histograms 
    !--------------------------------------------------------------------
    nnclass_hist = 0
    nncurve_hist = 0
    do i=1, natoms(frame)        
        if(nnclass(i)==1) nnclass_hist(1) = nnclass_hist(1) + 1
        if(nnclass(i)==2) nnclass_hist(2) = nnclass_hist(2) + 1 
        if(nnclass(i)==3) nnclass_hist(3) = nnclass_hist(3) + 1 
        if(nnclass(i)==4) nnclass_hist(4) = nnclass_hist(4) + 1
    end do
    
    
    do i=1, natoms(frame)
        if(flag_surf(i)==1) then
            nncurve_avg(i) = nncurve_avg(i) + 0.5_dp                    !shift halfway for 1deg digitization
            phis = int(nncurve_avg(i))
            if(phis>180) phis = 360-phis
            if(phis==0) phis = 1                                        !unrealistic small angles, avoid div0
            
            nncurve_hist(phis) = nncurve_hist(phis) + 1
        end if  
    end do
    
    
    
    !Write out files      
    
    !Write out surface classification   
    if(headers_done==0) then
        write(31,100)  
        100   format('SURFACE LAYER - Packing classification (class v no of particles)')
        write(31,110)
        110   format('      Frame',$)
        write(31,120)
        120   format(',        100',',        111',',        110',',        311')
    end if
    
    write(31,130) frame, nnclass_hist(1), nnclass_hist(2), nnclass_hist(3), nnclass_hist(4)
    130 format(i11,',',i11,',',i11,',',i11,',',i11) 
    
    !Write out to featureset file
    write(90,140) nnclass_hist(1), nnclass_hist(2), nnclass_hist(3), nnclass_hist(4)
    140 format(',',i11,',',i11,',',i11,',',i11,$)    
    
    
    
    !XYZ of surface classification
    write(34,'(i10,/)') natoms(frame)      
    do i=1, natoms(frame) 
        write(34,'(a,2x,3(F10.'//xyz_prec_str//',2X),i6)') lc(i),xc(i),yc(i),zc(i),nnclass(i)
    end do   
    
    
    
    !Write out surface curvature   
    if(headers_done==0) then
        write(32,200)  
        200   format('SURFACE LAYER - Curvature histogram (degrees v no. of particles)')
        write(32,210)
        210   format('      Frame',$)
        
        !-write out angle values  
        do i=1, 180
            write(32,220) i
            220     format(',',i11,$)
        end do
        
    end if
    
    write(32,*) ! new line
    
    !-write total angle label
    write(32,230) frame
    230 format(i11,$)      
    
    !-write total g3 out   
    do i=1, 180
        write(32,240) nncurve_hist(i)
        240   format(',',i11,$)      
    end do
    
    write(240,*) !new line      
    
    !-Write out to feature file
    do i=1, 180
        write(90,250) nncurve_hist(i)
        250   format(',',i11,$)      
    end do     
    
    
    !XYZ of surface curvature
    write(35,'(i10,/)') natoms(frame)      
    do i=1, natoms(frame) 
        write(35,'(a,2x,3(F10.'//xyz_prec_str//',2X),f10.1)') lc(i),xc(i),yc(i),zc(i),nncurve_avg(i)
    end do     
    

    
    
    DEALLOCATE(xbulk)
    DEALLOCATE(ybulk)
    DEALLOCATE(zbulk)
    DEALLOCATE(labatom)
    DEALLOCATE(nnclass)
    DEALLOCATE(nnangle)
    DEALLOCATE(nncurve)
    DEALLOCATE(nncurve_avg)
    
    
    return
end 


!END OF CAL_SURF_CURVE ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_SURF_XYZ ROUTINE
!Output sufrace layer (5th column - 0 bulk, 1 = surface)

subroutine CAL_SURF_XYZ
    
    use VARIABLES, only: flag_surf, frame, lc, natoms, xc, yc, zc, xyz_prec_str 
    IMPLICIT NONE
    
    
    !Local      
    integer :: i
    
    write(33,'(i10,/)') natoms(frame)       
    do i=1, natoms(frame)
        write(33,'(a,2x,3(F10.'//xyz_prec_str//',2X),i6)') lc(i),xc(i),yc(i),zc(i),flag_surf(i) 
    end do
    
    
    return
end 

!     END OF CAL_SURF_XYZ ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_CHAIN ROUTINE
!Calculate the number of atomic chain segments in a network
!using coordination C=2 only atoms (ignores dangling atoms C=1) 
!
!seg_co - segment atom (C=2) has 
!     = 0 (both neighbors lack C=2)
!     = 1 (one neighbor has C=2)
!     = 2 (both neighbors have C=2)
!seg_nn - atomic index of seg neighbors

subroutine CAL_CHAIN
    
    use VARIABLES, only: frame, headers_done, lc, natoms_max,           &
    natoms, ncord, nnlist, xc, yc, zc, xyz_prec_str
    
    IMPLICIT NONE

    
    !Local       
    integer :: i, j, it, it1, it2
    integer :: temp, count1, counter
    integer :: list_segments(natoms_max)
    integer :: seg_co(natoms_max)
    integer :: seg_nn(natoms_max,2)
    integer :: seg_id(natoms_max)
    integer :: change_flag, max_segid
    integer :: seg_histo(100000), flag_found
    integer :: seg_histogram(10000)
    
    integer :: labatom(natoms_max)      
    
    
    
    !Zero arrays
    seg_co = 0
    seg_id = 0     
    seg_histo = 0
    seg_histogram = 0
    
    !make a list of coordination 2 atoms
    count1 = 0
    do i=1, natoms(frame)
        
        temp = ncord(i)
        
        if(temp==2) then
            count1 = count1 + 1                                         !no. of coord 2 atoms
            list_segments(count1) = i                                   !index list of cord 2 atoms
        end if  
        
    end do  
    
        
    !make a list of neighbors of coordination 2 atoms
    do i=1, count1
        it = list_segments(i)                                           !it is atomic label
        
        do j=1, ncord(it)
            if(ncord(it)/=2) then
                print*,'ERROR!'                                         !should be coordination 2 atoms only
                STOP
            end if    
            it2 = nnlist(it,j)                                          !neighbor of cord 2 atom
            
            if(ncord(it2)==2) then                                      !if nn is also coord 2
                seg_co(it) = seg_co(it) + 1                             !can only be 0, 1 or 2 since nn is around only coord 2 atoms
                seg_nn(it,seg_co(it)) = it2                             !index of coord 2 atoms
            end if   
            
        end do  
    end do    
    
    !label C=2 segment atoms with the following rules
    !if isolated, give it a unique label
    !if one neighbor has a label, give it that label
    !if both neighbors have labels, give it the smallest label
    
    counter = 0       
    do i=1, count1                                                      !loop over cord 2 atoms
        it = list_segments(i)
        
        !for edge atom, label it if nn not label, else copy nn label        
        if(seg_co(it)==0) then
            seg_id(it) = -1  
        end if    
        
        
        if(seg_co(it)==1) then
            it1 = seg_nn(it,1) 
            if(seg_id(it1)==0) then
                counter = counter + 1
                seg_id(it) = counter
            end if  
            if(seg_id(it1)/=0) then
                seg_id(it) = seg_id(it1)  
            end if    
        end if
        
        
        if(seg_co(it)==2) then
            it1 = seg_nn(it,1)
            it2 = seg_nn(it,2)
            if(seg_id(it1)==0.and.seg_id(it2)==0) then
                counter = counter + 1
                seg_id(it) = counter
            end if    
            if(seg_id(it1)/=0.and.seg_id(it2)==0) then
                seg_id(it) = seg_id(it1)  
            end if 
            if(seg_id(it1)==0.and.seg_id(it2)/=0) then
                seg_id(it) = seg_id(it2)  
            end if 
            if(seg_id(it1)/=0.and.seg_id(it2)/=0) then
                if(seg_id(it1)>seg_id(it2)) then  
                    seg_id(it) = seg_id(it2)  
                end if
                if(seg_id(it1)<seg_id(it2)) then  
                    seg_id(it) = seg_id(it1)  
                end if
                if(seg_id(it1)==seg_id(it2)) then  
                    seg_id(it) = seg_id(it1)  
                end if
            end if     
        end if    
        
    end do  
    


    
    do   !infinite loop with exit condition
        change_flag = 0
        do i=1, count1
            it = list_segments(i)
            
            if(seg_id(it)/=-1) then
                
                if(seg_co(it)==1) then
                    it1 = seg_nn(it,1)  
                    if(seg_id(it)>seg_id(it1)) then
                        seg_id(it) = seg_id(it1)
                        change_flag = 1
                        EXIT
                    end if    
                end if   
                
                
                if(seg_co(it)==2) then
                    it1 = seg_nn(it,1)
                    it2 = seg_nn(it,2)
                    if(seg_id(it)>seg_id(it1).or.seg_id(it)>seg_id(it2)) then
                        seg_id(it) = MIN(seg_id(it1),seg_id(it2)) 
                        change_flag = 1
                        EXIT
                    end if  
                    
                    
                end if
                
                
            end if
        end do
        if(change_flag==0) EXIT
    end do
    
    !Maximum segment id     
    max_segid = -10
    do j=1, count1
        it = list_segments(j)
        if(seg_id(it)>max_segid) then
            max_segid = seg_id(it)  
        end if    
    end do
    
    !Make an ordered list of chains and how many member in each
    !-put all id=-1 (lone 2 folds) into first counter (seg_histo(1))
    do i=1, count1
        it = list_segments(i)  
        if(seg_id(it)==-1) then
            seg_histo(1) = seg_histo(1) + 1  
        end if  
    end do    
    
    counter = 1  !because counter=0 used by seg_id=-1
    do i=1, max_segid
        flag_found = 0
        do j=1, count1
            it = list_segments(j)
            
            if(i==seg_id(it)) then
                flag_found = 1  
            end if 
        end do
        
        if(flag_found==1) then
            counter = counter + 1
            do j=1, count1
                it = list_segments(j)
                
                if(seg_id(it)==i) then
                    seg_histo(counter) = seg_histo(counter) + 1  
                end if    
            end do  
        end if    
    end do  
    
    !make a histogram 
    seg_histogram(1) = seg_histo(1)
    do i=2, counter
        seg_histogram(seg_histo(i)) = seg_histogram(seg_histo(i)) + 1
    end do   
    
    
    
    
    
    !Output histogram
    if(headers_done==0) then
        write(28,110)
        110 format('CHAIN LENGTH HISTOGRAM') 
        write(28,120)
        120 format('      Frame',$)    
        do i=1, 20
            write(28,130) i
            130   format(',',i11,$)     
        end do  
        write(28,*) 
    end if
    
    
    write(28,140) frame 
    140 format(i11,$)     
    do i=1, 20
        write(28,150) seg_histogram(i)
        150   format(',',i11,$)       
    end do 
    write(28,*)
    
    
    
    
    !Write out to feature file
    do i=1, 20
        write(90,160) seg_histogram(i)
        160   format(',',i11,$)       
    end do             
    
    
    !Output XYZ file
    
    !Make labels
    !0 = unclassifed
    !1 = C=2 particle with 0 C=2 neighbors 
    !2 = C=2 particle with 1 C=2 neighbors
    !3 = C=2 particle with 2 C=2 neighbors 
    
    labatom = 0 !zero array  
    do i=1, natoms(frame)
        if(seg_co(i)==0.and.ncord(i)==2) then
            labatom(i) = 1
        end if  
        if (seg_co(i)==1) then
            labatom(i) = 2
        end if  
        if (seg_co(i)==2) then
            labatom(i) = 3
        end if
    end do
    
    
    !Write out to file     
    write(29,'(i10,/)') natoms(frame)        
    do i=1, natoms(frame)
        write(29,'(a,2x,3(F10.'//xyz_prec_str//',2X),i6)') lc(i),xc(i),yc(i),zc(i),labatom(i) 
    end do  
    
    
    return
end 


!END OF CAL_CHAIN ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_COORD ROUTINE
!Calculate the coordination statistics for total particles, 
!Type 1, 2, 3... for bulk and surface

subroutine CAL_COORD
    
    call CAL_COORD_HISTO       
    call CAL_COORD_XYZ
    
    return
end 


!END OF CAL_COORD ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_COORD_HISTO ROUTINE
!Calculate various coordination (N) statistics for all particles
!     
!Coordination histograms calculated for ALL, BULK, SURFACE and SURFACE LAYER ONLY particles 
!There later three require a surface layer.  SURFACE coordination is the usual coordination
!of the surface particle.  SURFACE ONLY coordination is the surface particle coordination
!using only particle on the surface layer (not the bulk atoms below like in SURFACE coordination).
!
!coord_all(M,N)   - ALL particle coordination histogram 
!coord_bulk(M,N)  - BULK particle coordination histogram
!coord_surf(M,N)  - SURF particle coordination histogram
!coord_surfo(M,N) - SURF LAYER ONLY particle coordination histogram
!where M is the particle type and N is the coordination number bin for the histogram. 
!M is 1-10 for up to ten elements (partial coordination) and 11 is reserve for the total coordination histogram
!N goes to 1001 instead of 1000 because coordination bins shifts by 1 to ensure zero coordinated bin array works 

subroutine CAL_COORD_HISTO
    
    use KINDS, only: dp
    use VARIABLES, only: c_lab, flag_surf,frame, headers_done,         &
    natoms, ncord, gcn, ncordS, gcnS, typecount, typelist,             &
    nnlist, nnlistS 
    IMPLICIT NONE

    
    !Local 
    integer :: i, j
    integer :: plabel, pcoord, pcoordo, pgcn, pgcno
    integer :: counter_all(11)
    integer :: counter_bulk(11)
    integer :: counter_surf(11)
    integer :: counter_surfo(11)
    integer :: gcn_counter_all(11)
    integer :: gcn_counter_bulk(11)
    integer :: gcn_counter_surf(11)
    integer :: gcn_counter_surfo(11)
    
    integer :: coord_all(11,1001)                                  
    integer :: coord_bulk(11,1001)
    integer :: coord_surf(11,1001)                                         
    integer :: coord_surfo(11,1001)      
    integer :: gcn_all(11,1001)                                  
    integer :: gcn_bulk(11,1001)
    integer :: gcn_surf(11,1001)                                         
    integer :: gcn_surfo(11,1001)      
    
    real(dp) :: coord_avg_all(11)        !avg coord of all particles     
    real(dp) :: coord_avg_bulk(11)       !avg coord of bulk particles
    real(dp) :: coord_avg_surf(11)       !avg coord of surf particles
    real(dp) :: coord_avg_surfo(11)      !avg coord of surf particles using only surface layer
    real(dp) :: gcn_avg_all(11)
    real(dp) :: gcn_avg_bulk(11)
    real(dp) :: gcn_avg_surf(11)
    real(dp) :: gcn_avg_surfo(11)
    
    
    
    !zero histograms      
    coord_all   = 0
    coord_bulk  = 0
    coord_surf  = 0
    coord_surfo = 0
    gcn_all     = 0
    gcn_bulk    = 0
    gcn_surf    = 0
    gcn_surfo   = 0
    
    !zero counters      
    coord_avg_all   = 0.0_dp
    coord_avg_bulk  = 0.0_dp
    coord_avg_surf  = 0.0_dp
    coord_avg_surfo = 0.0_dp  
    gcn_avg_all     = 0.0_dp
    gcn_avg_bulk    = 0.0_dp
    gcn_avg_surf    = 0.0_dp
    gcn_avg_surfo   = 0.0_dp  
    
    counter_all       = 0
    counter_bulk      = 0
    counter_surf      = 0
    counter_surfo     = 0
    gcn_counter_all   = 0
    gcn_counter_bulk  = 0
    gcn_counter_surf  = 0
    gcn_counter_surfo = 0
    
    
    do i=1, natoms(frame)                                               !loop over particles
        
        plabel  = c_lab(i)                                              !particle type of particle i
        pcoord  = ncord(i) + 1                                          !+1 to shift to avoid zero coord issue
        pcoordo = ncordS(i) + 1                                             
        pgcn    = ceiling(gcn(i)) + 1
        pgcno   = ceiling(gcnS(i)) + 1                                             
        
        !Check for exceeding total coordination  
        if(pcoord>1001) then
            print*,'ERROR: More than 1000 nearest neighbors detected (total coordination)'
            STOP
        end if 

        !Check for exceeding surface coordination         
        if(flag_surf(i)==1.and.pcoordo>1001) then
            print*,'ERROR: More than 1000 nearest neighbors detected (surface coordination)'
            STOP        
        end if
        
        !Total coordination 
        coord_all(plabel,pcoord)       = coord_all(plabel,pcoord)       + 1  
        coord_all(11,pcoord)           = coord_all(11,pcoord)           + 1

        gcn_all(plabel,pgcn)           = gcn_all(plabel,pgcn)           + 1  
        gcn_all(11,pgcn)               = gcn_all(11,pgcn)               + 1

        gcn_avg_all(plabel)            = gcn_avg_all(plabel)            + gcn(i)
        gcn_avg_all(11)                = gcn_avg_all(11)                + gcn(i)
        gcn_counter_all(plabel)        = gcn_counter_all(plabel)        + 1  
        gcn_counter_all(11)            = gcn_counter_all(11)            + 1
        
        !Bulk coordination
        if(flag_surf(i)==0) then
            coord_bulk(plabel,pcoord)    = coord_bulk(plabel,pcoord) + 1    
            coord_bulk(11,pcoord)        = coord_bulk(11,pcoord)     + 1  

            gcn_bulk(plabel,pgcn)        = gcn_bulk(plabel,pgcn)     + 1    
            gcn_bulk(11,pgcn)            = gcn_bulk(11,pgcn)         + 1  

            gcn_avg_bulk(plabel)         = gcn_avg_bulk(plabel)      + gcn(i)
            gcn_avg_bulk(11)             = gcn_avg_bulk(11)          + gcn(i)
            gcn_counter_bulk(plabel)     = gcn_counter_bulk(plabel)  + 1
            gcn_counter_bulk(11)         = gcn_counter_bulk(11)      + 1
        end if 
        
        !Surface and surface only coordination         
        if(flag_surf(i)==1) then
            coord_surf(plabel,pcoord)    = coord_surf(plabel,pcoord)   + 1
            coord_surf(11,pcoord)        = coord_surf(11,pcoord)       + 1
            coord_surfo(plabel,pcoordo)  = coord_surfo(plabel,pcoordo) + 1
            coord_surfo(11,pcoordo)      = coord_surfo(11,pcoordo)     + 1 

            gcn_surf(plabel,pgcn)        = gcn_surf(plabel,pgcn)       + 1
            gcn_surf(11,pgcn)            = gcn_surf(11,pgcn)           + 1
            gcn_surfo(plabel,pgcno)      = gcn_surfo(plabel,pgcno)     + 1
            gcn_surfo(11,pgcno)          = gcn_surfo(11,pgcno)         + 1 

            gcn_avg_surf(plabel)         = gcn_avg_surf(plabel)        + gcn(i)
            gcn_avg_surf(11)             = gcn_avg_surf(11)            + gcn(i)
            gcn_counter_surf(plabel)     = gcn_counter_surf(plabel)    + 1
            gcn_counter_surf(11)         = gcn_counter_surf(11)        + 1
            gcn_avg_surfo(plabel)        = gcn_avg_surfo(plabel)       + gcnS(i)
            gcn_avg_surfo(11)            = gcn_avg_surfo(11)           + gcnS(i)
            gcn_counter_surfo(plabel)    = gcn_counter_surfo(plabel)   + 1
            gcn_counter_surfo(11)        = gcn_counter_surfo(11)       + 1
        end if
        
    end do  
    
    
    !Average coordinations
    do i=1, typecount
        do j=1, 1001
            
            !partial coordination for type i        
            coord_avg_all(i)    = coord_avg_all(i)    + coord_all(i,j)  * (j-1)   !j-1 for starting at 1 
            coord_avg_bulk(i)   = coord_avg_bulk(i)   + coord_bulk(i,j) * (j-1) 
            coord_avg_surf(i)   = coord_avg_surf(i)   + coord_surf(i,j) * (j-1)   
            coord_avg_surfo(i)  = coord_avg_surfo(i)  + coord_surfo(i,j)* (j-1)  
            
            counter_all(i)   = counter_all(i)   + coord_all(i,j)
            counter_bulk(i)  = counter_bulk(i)  + coord_bulk(i,j)
            counter_surf(i)  = counter_surf(i)  + coord_surf(i,j)
            counter_surfo(i) = counter_surfo(i) + coord_surfo(i,j)
            
            !total coordination           
            coord_avg_all(11)   = coord_avg_all(11)   + coord_all(i,j)   * (j-1)   
            coord_avg_bulk(11)  = coord_avg_bulk(11)  + coord_bulk(i,j)  * (j-1) 
            coord_avg_surf(11)  = coord_avg_surf(11)  + coord_surf(i,j)  * (j-1)   
            coord_avg_surfo(11) = coord_avg_surfo(11) + coord_surfo(i,j) * (j-1)   
            
            counter_all(11)   = counter_all(11)   + coord_all(i,j)
            counter_bulk(11)  = counter_bulk(11)  + coord_bulk(i,j)
            counter_surf(11)  = counter_surf(11)  + coord_surf(i,j)
            counter_surfo(11) = counter_surfo(11) + coord_surfo(i,j)
            
        end do  
    end do
    
    do i=1,11
        coord_avg_all(i)   = coord_avg_all(i)   / counter_all(i)
        coord_avg_bulk(i)  = coord_avg_bulk(i)  / counter_bulk(i)
        coord_avg_surf(i)  = coord_avg_surf(i)  / counter_surf(i)
        coord_avg_surfo(i) = coord_avg_surfo(i) / counter_surfo(i)
        gcn_avg_all(i)   = gcn_avg_all(i)   / gcn_counter_all(i)
        gcn_avg_bulk(i)  = gcn_avg_bulk(i)  / gcn_counter_bulk(i)
        gcn_avg_surf(i)  = gcn_avg_surf(i)  / gcn_counter_surf(i)
        gcn_avg_surfo(i) = gcn_avg_surfo(i) / gcn_counter_surfo(i)
    end do
    
    
    
    !Write out coordination statistics
    
    if(headers_done==0) then
        write(20,110)
        110 format('COORDINATION STATS (T=total B=bulk S=surface So=surface only G=generalised coordination number)')  
        write(20,120)
        120 format('      Frame',',       Type'$)
        write(20,130)
        130 format(                                                    &
         ',  Avg Total',',   Avg Bulk',',   Avg Surf',',  Avg Surfo'   &
        ,', Avg GTotal',',  Avg GBulk',',  Avg GSurf',', Avg GSurfo'   &
        ,',         T0',',         T1',',         T2',',         T3'   &
        ,',         T4',',         T5',',         T6',',         T7'   &
        ,',         T8',',         T9',',        T10',',        T11'   &
        ,',        T12',',        T13',',        T14',',        T15'   &
        ,',        T16',',        T17',',        T18',',        T19'   & 
        ,',        T20'                                                &
        
        ,',         B0',',         B1',',         B2',',         B3'   &
        ,',         B4',',         B5',',         B6',',         B7'   &
        ,',         B8',',         B9',',        B10',',        B11'   &
        ,',        B12',',        B13',',        B14',',        B15'   &
        ,',        B16',',        B17',',        B18',',        B19'   & 
        ,',        B20'                                                &
        
        ,',         S0',',         S1',',         S2',',         S3'   &
        ,',         S4',',         S5',',         S6',',         S7'   &
        ,',         S8',',         S9',',        S10',',        S11'   &
        ,',        S12',',        S13',',        S14',',        S15'   &
        ,',        S16',',        S17',',        S18',',        S19'   & 
        ,',        S20'                                                &

        ,',        S0o',',        S1o',',        S2o',',        S3o'   &
        ,',        S4o',',        S5o',',        S6o',',        S7o'   &
        ,',        S8o',',        S9o',',       S10o',',       S11o'   &
        ,',       S12o',',       S13o',',       S14o',',       S15o'   &
        ,',       S16o',',       S17o',',       S18o',',       S19o'   & 
        ,',       S20o'                                                &
 
        ,',        GT0',',        GT1',',        GT2',',        GT3'   &
        ,',        GT4',',        GT5',',        GT6',',        GT7'   &
        ,',        GT8',',        GT9',',       GT10',',       GT11'   &
        ,',       GT12',',       GT13',',       GT14',',       GT15'   &
        ,',       GT16',',       GT17',',       GT18',',       GT19'   & 
        ,',       GT20'                                                &
        
        ,',        GB0',',        GB1',',        GB2',',        GB3'   &
        ,',        GB4',',        GB5',',        GB6',',        GB7'   &
        ,',        GB8',',        GB9',',       GB10',',       GB11'   &
        ,',       GB12',',       GB13',',       GB14',',       GB15'   &
        ,',       GB16',',       GB17',',       GB18',',       GB19'   & 
        ,',       GB20'                                                &
        
        ,',        GS0',',        GS1',',        GS2',',        GS3'   &
        ,',        GS4',',        GS5',',        GS6',',        GS7'   &
        ,',        GS8',',        GS9',',       GS10',',       GS11'   &
        ,',       GS12',',       GS13',',       GS14',',       GS15'   &
        ,',       GS16',',       GS17',',       GS18',',       GS19'   & 
        ,',       GS20'                                                &

        ,',       GS0o',',       GS1o',',       GS2o',',       GS3o'   &
        ,',       GS4o',',       GS5o',',       GS6o',',       GS7o'   &
        ,',       GS8o',',       GS9o',',      GS10o',',      GS11o'   &
        ,',      GS12o',',      GS13o',',      GS14o',',      GS15o'   &
        ,',      GS16o',',      GS17o',',      GS18o',',      GS19o'   & 
        ,',      GS20o')                                               
    end if
    
    
    !   -Output ALL stats
    write(20,145) frame,'      Total',coord_avg_all(11),                 &!ALL Average coord stats
    coord_avg_bulk(11),coord_avg_surf(11),coord_avg_surfo(11),           &
    gcn_avg_all(11),gcn_avg_bulk(11),gcn_avg_surf(11),gcn_avg_surfo(11)   !ALL Average gcn stats  
    145       format(i11,',',a11,8(',',f11.6),$)  
    
    write(90,146) '      Total',coord_avg_all(11),coord_avg_bulk(11),  &!FEATUREFILE
    coord_avg_surf(11),coord_avg_surfo(11),gcn_avg_all(11),            &
    gcn_avg_bulk(11),gcn_avg_surf(11),gcn_avg_surfo(11)
    146       format(',',a11,8(',',f11.6),$) 
    

    do j=1, 21                                                          !ALL total coord dist
        write(20,150) coord_all(11,j)
        150       format(',',i11,$)      
    end do    
    do j=1, 21                                                          !feature file             
        write(90,151) coord_all(11,j)
        151       format(',',i11,$)      
    end do
    
    
    do j=1, 21                                                          !ALL bulk coord dist
        write(20,152) coord_bulk(11,j)
        152      format(',',i11,$)      
    end do
    do j=1, 21                                                          !feature file 
        write(90,153) coord_bulk(11,j)
        153      format(',',i11,$)      
    end do
    
    
    do j=1, 21                                                          !ALL surf coord dist
        write(20,154) coord_surf(11,j)
        154       format(',',i11,$)      
    end do
    do j=1, 21                                                          !feature file 
        write(90,155) coord_surf(11,j)
        155       format(',',i11,$)      
    end do     
         
         
    do j=1, 21                                                          !ALL surfo coord dist
        write(20,156) coord_surfo(11,j)
        156       format(',',i11,$)      
    end do  
    do j=1, 21                                                          !feature file 
        write(90,157) coord_surfo(11,j)
        157       format(',',i11,$)      
    end do     



    do j=1, 21                                                          !ALL total gcn dist
        write(20,170) gcn_all(11,j)
        170       format(',',i11,$)      
    end do    
    do j=1, 21                                                          !feature file             
        write(90,171) gcn_all(11,j)
        171       format(',',i11,$)      
    end do
    
    
    do j=1, 21                                                          !ALL bulk gcn dist
        write(20,172) gcn_bulk(11,j)
        172      format(',',i11,$)      
    end do
    do j=1, 21                                                          !feature file 
        write(90,173) gcn_bulk(11,j)
        173      format(',',i11,$)      
    end do
    
    
    do j=1, 21                                                          !ALL surf gcn dist
        write(20,174) gcn_surf(11,j)
        174       format(',',i11,$)      
    end do
    do j=1, 21                                                          !feature file 
        write(90,175) gcn_surf(11,j)
        175       format(',',i11,$)      
    end do     
         
         
    do j=1, 21                                                          !ALL surfo gcn dist
        write(20,176) gcn_surfo(11,j)
        176       format(',',i11,$)      
    end do  
    do j=1, 21                                                          !feature file 
        write(90,177) gcn_surfo(11,j)
        177       format(',',i11,$)      
    end do     
    
    write(20,*)  !New line


!   -Output Partial stats    
    if(typecount>1) then
        do i=1, typecount


        
            write(20,180) frame,i,coord_avg_all(i),coord_avg_bulk(i),  &!Partial Average coord stats
            coord_avg_surf(i),coord_avg_surfo(i)                               
            180       format(i11,',',(9x,i2),4(',',f11.6),$)   

            write(90,181) i,coord_avg_all(i),coord_avg_bulk(i),        &!feature file
            coord_avg_surf(i),coord_avg_surfo(i)                               
            181       format(',',i11,',',f11.6,',',f11.6,',',f11.6,',',f11.6,$) 

            write(20,200) gcn_avg_all(i),gcn_avg_bulk(i),  &!Partial Average gcn stats
            gcn_avg_surf(i),gcn_avg_surfo(i)                               
            200       format(4(',',f11.6),$)   

            write(90,201) gcn_avg_all(i),gcn_avg_bulk(i),        &!feature file
            gcn_avg_surf(i),gcn_avg_surfo(i)                               
            201       format(',',f11.6,',',f11.6,',',f11.6,',',f11.6,$) 
            


            do j=1, 21                                                  !Partial total coord dist
                write(20,185) coord_all(i,j)
                185       format(',',i11,$)      
            end do    
            do j=1, 21                                                  !feature file             
                write(90,186) coord_all(i,j)
                186       format(',',i11,$)      
            end do
    
    
            do j=1, 21                                                  !Partial bulk coord dist
                write(20,187) coord_bulk(i,j)
                187      format(',',i11,$)      
            end do
            do j=1, 21                                                  !feature file 
                write(90,188) coord_bulk(i,j)
                188      format(',',i11,$)      
            end do
    
    
            do j=1, 21                                                  !Partial surf coord dist
                write(20,189) coord_surf(i,j)
                189       format(',',i11,$)      
            end do
            do j=1, 21                                                  !feature file 
                write(90,190) coord_surf(i,j)
                190       format(',',i11,$)      
            end do     
        
         
            do j=1, 21                                                  !Partial surfo coord dist
                write(20,191) coord_surfo(i,j)
                191       format(',',i11,$)      
            end do  
            do j=1, 21                                                  !feature file 
                write(90,192) coord_surfo(i,j)
                192       format(',',i11,$)      
            end do  
            


            
            do j=1, 21                                                  !Partial total gcn dist
                write(20,205) gcn_all(i,j)
                205       format(',',i11,$)      
            end do    
            do j=1, 21                                                  !feature file             
                write(90,206) gcn_all(i,j)
                206       format(',',i11,$)      
            end do
    
    
            do j=1, 21                                                  !Partial bulk gcn dist
                write(20,207) gcn_bulk(i,j)
                207      format(',',i11,$)      
            end do
            do j=1, 21                                                  !feature file 
                write(90,208) gcn_bulk(i,j)
                208      format(',',i11,$)      
            end do
    
    
            do j=1, 21                                                  !Partial surf gcn dist
                write(20,209) gcn_surf(i,j)
                209       format(',',i11,$)      
            end do
            do j=1, 21                                                  !feature file 
                write(90,210) gcn_surf(i,j)
                210       format(',',i11,$)      
            end do     
        
         
            do j=1, 21                                                  !Partial surfo gcn dist
                write(20,211) gcn_surfo(i,j)
                211       format(',',i11,$)      
            end do  
            do j=1, 21                                                  !feature file 
                write(90,212) gcn_surfo(i,j)
                212       format(',',i11,$)      
            end do  
            
            write(20,*)  !New line
        end do
    end if
    

    return
end 


!END OF CAL_COORD_HISTO ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_COORD_XYZ ROUTINE
!Output particle coordination for visualization 
!Surface coordination only assigns a coordination to surface particles
!using only other surface particles as nearest neighbors (bulk particles
!a equal to zero

subroutine CAL_COORD_XYZ
    
    use VARIABLES, only: frame, natoms, lc, ncord, ncordA, gcn,        &
    gcnA, ncordS, ncordSA, gcnS, gcnSA, xc, yc, zc, xyz_prec_str,      &
    typecount 
    IMPLICIT NONE
    
     
    
    !Local 
    integer :: i,j,k
    
    !Write out xyz file
    !-Total coordination
    write(21,'(i10,/)') natoms(frame)        
    do i=1, natoms(frame)
        write(21,'(a,2x,3(F10.'//xyz_prec_str//',2X),i6,F10.1,2X)',advance='no') lc(i),xc(i),yc(i),zc(i),ncord(i),gcn(i)
        do j=1, typecount
            do k=j, typecount
                write(21,'(i6)',advance='no') ncordA(i,j,k)
            end do
        end do
        do j=1, typecount
            do k=j, typecount
                write(21,'(F10.1,2X)',advance='no') gcnA(i,j,k)
            end do
        end do
        write(21,*)  !New line
    end do  
    
    !-Surface coordination
    write(22,'(i10,/)') natoms(frame)  
    do i=1, natoms(frame)
        write(22,'(a,2x,3(F10.'//xyz_prec_str//',2X),i6,F10.1,2X)',advance='no') lc(i),xc(i),yc(i),zc(i),ncordS(i),gcnS(i)
        do j=1, typecount
            do k=j, typecount
                write(22,'(i6)',advance='no') ncordSA(i,j,k)
            end do
        end do
        do j=1, typecount
            do k=j, typecount
                write(22,'(F10.1,2X)',advance='no') gcnSA(i,j,k)
            end do
        end do
        write(22,*)  !New line
    end do        
    
    return
end 


!END OF CAL_COORD_XYZ ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_BLENGTH ROUTINE
!Calculate the average bond lengths and std. dev. for particle types

subroutine CAL_BLENGTH
    
    use KINDS, only: dp
    use VARIABLES, only: c_lab, con_maxnn, frame, headers_done, lc,    &
    natoms_max, natoms, ncord, nnlist, typecount,                      &
    mid_x, mid_y, mid_z,                                               &
    xc, yc, zc, xl, yl, zl, xl2inv, yl2inv, zl2inv,                    &
    in_xyz_prec, xyz_prec_str 
    IMPLICIT NONE
    
    
    !Local variables  
    integer :: i,j
    integer :: it, it1, it2
    integer :: bondhist(typecount,typecount)                            !bond type histogram
    integer :: bondhist_T                                               !same as above but for all particles    
    integer :: cnt(natoms_max)                                       !number of bonds for each atom
    real(dp) :: dx, dy, dz
    real(dp) :: dr2
    real(dp) :: b_dr, b_dr2                                             !difference between mean and value, square of 
    real(dp) :: b_dr_T, b_dr2_T
    real(dp) :: b_dr2sum(typecount,typecount)                           !sum of diff. btw mean and value squared
    real(dp) :: b_dr2sum_T
    real(dp) :: b_variance(typecount,typecount)                         !variance of bond lengths
    real(dp) :: b_variance_T
    real(dp) :: b_standdev(typecount,typecount)                         !standard deviation of bond lengths 
    real(dp) :: b_standdev_T
    real(dp) :: blengthsum(typecount,typecount)                         !sum of distance of between type i and j
    real(dp) :: blengthsum_T        
    real(dp) :: blengthavg(typecount,typecount)                         !average bond length between type i and j  
    real(dp) :: blengthavg_T    
    real(dp) :: blengthmax(typecount,typecount)
    real(dp) :: blengthmax_T    
    real(dp) :: blengthmin(typecount,typecount)
    real(dp) :: blengthmin_T    
    real(dp), allocatable :: blength(:,:)                               !bond length between atom i and j
    real(dp) :: blength_pavg(natoms_max)
    real(dp) :: blength_pmax(natoms_max)
    real(dp) :: blength_pmin(natoms_max)
    real(dp) :: rad_dist(natoms_max)
    integer :: blen_prec
    integer :: rad_prec
    character(len=50) :: blen_prec_str
    character(len=50) :: rad_prec_str
    character(len=200) :: writeblen
    character(len=200) :: writerad
 
    
    !Allocate array      
    ALLOCATE(blength(natoms_max,con_maxnn))      
    
    
    !Zero arrays  
    blength = 0  
    bondhist = 0
    bondhist_T = 0
    cnt = 0
    blengthsum = 0.0_dp
    blengthsum_T = 0.0_dp
    blengthmax = 0.0_dp
    blengthmax_T = 0.0_dp
    blengthmin = 0.0_dp
    blengthmin_T = 0.0_dp
    
    
    !Calculate bond length array      
    !-build neighbor list
    blength_pavg = 0.0_dp
    blength_pmax = 0.0_dp
    blength_pmin = 10.0_dp
    do i = 1, natoms(frame)
        do j = 1, ncord(i)
             
            it = nnlist(i,j)
            
            dx = xc(i) - xc(it)
            dy = yc(i) - yc(it)
            dz = zc(i) - zc(it)
            
            dx = dx -(int(dx*xl2inv)*xl)
            dy = dy -(int(dy*yl2inv)*yl)
            dz = dz -(int(dz*zl2inv)*zl)
            
            dr2 = dx*dx + dy*dy + dz*dz
            
            blength(i,j) = dsqrt(dr2)                                   !store nn distances
            cnt(i) = cnt(i) + 1
            blength_pavg(i) = blength_pavg(i) + blength(i,j)
            !if(ncord(i)==4) then
            !    print*,i,xc(i),yc(i),zc(i)
            !    print*,it,xc(it),yc(it),zc(it)
            !    print*,it,dx,dy,dz,blength(i,j)
            !end if
            !if(i==716) then
            !    print*,i,blength(i,j),it
            !elseif(i==721) then
            !    print*,i,blength(i,j),it
            !end if
            if (blength(i,j)>blength_pmax(i)) then
                blength_pmax(i) = blength(i,j)
            end if
            if (blength(i,j)<blength_pmin(i)) then
                blength_pmin(i) = blength(i,j)
            end if
        end do

        if(ncord(i)>0) then
            blength_pavg(i) = blength_pavg(i) / ncord(i)                !average bond length from particle i
        end if

        rad_dist(i) = sqrt((xc(i)-mid_x)**2 + (yc(i)-mid_y)**2 + (zc(i)-mid_z)**2)
    end do       
    
    
    
    !Write out xyz file
    blen_prec = in_xyz_prec - 1
    write(blen_prec_str,*) blen_prec 
    writeblen = '(a,2x,3(F10.'//xyz_prec_str//',2X),3(F10.'//blen_prec_str//',2X),i6)'
    write(27,'(i10,/)') natoms(frame)        
    do i=1, natoms(frame)
        write(27,writeblen) lc(i),xc(i),yc(i),zc(i),blength_pavg(i),blength_pmax(i),blength_pmin(i),cnt(i)
    end do  
    
    rad_prec = in_xyz_prec - 2
    write(rad_prec_str,*) rad_prec 
    writerad = '(a,2x,3(F10.'//xyz_prec_str//',2X),F10.'//rad_prec_str//')'
    write(59,'(i10,/)') natoms(frame)        
    do i=1, natoms(frame)
        write(59,writerad) lc(i),xc(i),yc(i),zc(i),rad_dist(i)
    end do  
    
    
    !Calculate distribution of bond types and sum bondlengths, max and mins 
    do i=1, natoms(frame)
        do j=1, ncord(i)
            it = nnlist(i,j) 
            
            do it1=1,typecount
                do it2=it1,typecount            
                    if(c_lab(i)==it1.and.c_lab(it)==it2) then
                        blengthsum(it1,it2) = blengthsum(it1,it2) + blength(i,j) 
                        bondhist(it1,it2) = bondhist(it1,it2) + 1  
                        
                        if(bondhist(it1,it2)==1) then         !initialize max/min of 1st value
                            blengthmax(it1,it2) = blength(i,j)
                            blengthmin(it1,it2) = blength(i,j)
                        end if
                        if(blength(i,j)<blengthmin(it1,it2)) blengthmin(it1,it2) = blength(i,j)
                        if(blength(i,j)>blengthmax(it1,it2)) blengthmax(it1,it2) = blength(i,j)
                    end if 
   
                end do
            end do  
            
            !Total stats for all particles                
            blengthsum_T = blengthsum_T + blength(i,j)
            bondhist_T = bondhist_T + 1
            if(bondhist_T==1) then
                blengthmax_T = blength(i,j)
                blengthmin_T = blength(i,j)
            end if
            if(blength(i,j)<blengthmin_T) blengthmin_T = blength(i,j)
            if(blength(i,j)>blengthmax_T) blengthmax_T = blength(i,j) 
            
        end do
    end do
    

    
    
    !1-2=2-1      
    if(typecount>1) then
        do i=1, typecount                  
            do j=i+1, typecount
                blengthsum(j,i) = blengthsum(i,j)
                bondhist(j,i) = bondhist(i,j)
            end do
        end do 
    end if
    
    
    
    !half non-diagnals counts for double counting
    do i=1,typecount
        blengthsum(i,i) = blengthsum(i,i) / 2.0_dp
        bondhist(i,i) = bondhist(i,i) / 2      
    end do 
    blengthsum_T = blengthsum_T / 2.0_dp
    bondhist_T = bondhist_T / 2
    
    

    !Average bond lengths   
    do i=1, typecount
        do j=i, typecount   
            if(bondhist(i,j)/=0) then            
                blengthavg(i,j) = blengthsum(i,j) / bondhist(i,j)
            else
                blengthavg(i,j) = 0.0_dp
            end if        
        end do
    end do 
    if(bondhist_T/=0) then
        blengthavg_T = blengthsum_T / bondhist_T  
    else
        blengthavg_T = 0.0_dp
    end if
    
    
    
    !bond length 1-2 = 2-1
    do i=1, typecount
        do j=i+1, typecount
            blengthavg(j,i) = blengthavg(i,j)
        end do
    end do  
    
    
    !Standard Deviation in bond lengths
    b_dr2sum = 0.0_dp
    b_dr2sum_T = 0.0_dp
    do i=1, natoms(frame)      
        do j=1, ncord(i)
            it = nnlist(i,j)
            
            do it1=1,typecount
                do it2=it1,typecount              
                    if(c_lab(i)==it1.and.c_lab(it)==it2) then
                        b_dr = blength(i,j) - blengthavg(it1,it2)
                        b_dr2 = b_dr * b_dr
                        if(b_dr2<1.0e-20_dp) b_dr2 = 0.0_dp             !avoid small number warning
                        b_dr2sum(it1,it2) = b_dr2sum(it1,it2) + b_dr2   
                    end if 
                end do
            end do  
            
            !Total stats for all particles        
            b_dr_T = blength(i,j) - blengthavg_T
            b_dr2_T = b_dr_T * b_dr_T
            if(b_dr2_T<1.0e-20_dp) b_dr2_T = 0.0_dp                     !avoid small number warning
            b_dr2sum_T = b_dr2sum_T + b_dr2_T
            
        end do
    end do 
    
    
    !1-2=2-1      
    do i=1, typecount
        do j=i+1, typecount
            b_dr2sum(j,i) = b_dr2sum(i,j)
        end do
    end do  
    
    !Half non-diagnals counts for double counting
    do i=1,typecount    
        b_dr2sum(i,i) = b_dr2sum(i,i) / 2.0_dp
    end do   
    b_dr2sum_T = b_dr2sum_T / 2.0_dp
    
    
    do i=1, typecount
        do j=i, typecount
            if(bondhist(i,j)/=0) then
                b_variance(i,j) = b_dr2sum(i,j) / bondhist(i,j)
                b_standdev(i,j) = sqrt(b_variance(i,j))
            else
                b_variance(i,j) = 0.0_dp
                b_standdev(i,j) = 0.0_dp
            end if          
        end do
    end do  
    if(bondhist_T/=0) then
        b_variance_T = b_dr2sum_T / bondhist_T
        b_standdev_T = sqrt(b_variance_T)
    else
        b_variance_T = 0.0_dp
        b_standdev_T = 0.0_dp
    end if    
    
    
    !Deallocate
    DEALLOCATE(blength)
    
    
    
    !Write out data
    if(headers_done==0) then
        write(26,110)
        110 format('BOND LENGTH STATS (Column blocks for type-type combinations (lengths in Angstrom))') 
        
        write(26,115)
        115 format('      Frame',$)
        
        !Total particles
        write(26,117)
        117 format(',      Type1',',      Type2',', Avg Length', &
                   ',    Std Dev',', Max Length',', Min Length',',  N Lengths',$)
        
        
        !Partial particles
        do i=1, typecount
            do j=i, typecount
                write(26,120)
                120 format(',      Type1',',      Type2',', Avg Length', &
                           ',    Std Dev',', Max Length',', Min Length',',  N Lengths',$)    
            end do
        end do
            
        write(26,*)                                                     !new line 
    end if
    
    
    write(26,140) frame 
    140 format(i11,$) 
    
    !Total particles
    write(26,142) blengthavg_T,b_standdev_T,blengthmax_T,blengthmin_T,bondhist_T
    142     format(',      Total',',      Total',',',f8.3,',',f8.3,',',f8.3,',',f8.3,',',i8,$)     
    !         -Write out to feature file
    write(90,144) blengthavg_T,b_standdev_T,blengthmax_T,blengthmin_T,bondhist_T
    144     format(',      Total',',      Total',',',f11.3,',',f11.3,',',f11.3,',',f11.3,',',i11,$) 
    
    !Partial particles
    do i=1, typecount
        do j=i, typecount
            write(26,150) i,j,blengthavg(i,j),b_standdev(i,j),blengthmax(i,j),blengthmin(i,j),bondhist(i,j)
            150     format(',',(9x,i2),',',(9x,i2),',',f8.3,',',f8.3,',',f8.3,',',f8.3,',',i8,$)     
            
            !         -Write out to feature file
            write(90,151) i,j,blengthavg(i,j),b_standdev(i,j),blengthmax(i,j),blengthmin(i,j),bondhist(i,j)
            151     format(',',(9x,i2),',',(9x,i2),',',f11.3,',',f11.3,',',f11.3,',',f11.3,',',i11.6,$)    
            
            
        end do
    end do 
    write(26,*)
    
    
    return
end 


!END OF CAL_BLENGTH ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_BTYPE ROUTINE
!Calculate distribution of bond types 

subroutine CAL_BTYPE
    
    use KINDS, only: dp
    use VARIABLES, only: c_lab, frame, headers_done,                   &
    natoms, ncord, nnlist, typecount    
    IMPLICIT NONE
       
    
    !Local 
    integer :: i, j
    integer :: it, it1, it2
    real(dp), allocatable :: bondhist(:,:)           !no. bonds between types i and j
    real(dp), allocatable :: bondhist_fraction(:,:)  !fraction of bonds between types i and j
    real(dp) bondhist_sum                            !total number of bonds (no double counting)
    
    
    ALLOCATE(bondhist(typecount,typecount))
    ALLOCATE(bondhist_fraction(typecount,typecount))
    
    bondhist = 0
    bondhist_fraction = 0.0_dp
    bondhist_sum = 0.0_dp 
    
    
    do i=1, natoms(frame)
        do j=1, ncord(i)
            it = nnlist(i,j) 
            
            do it1=1,typecount
                do it2=it1,typecount
                    if(c_lab(i)==it1.and.c_lab(it)==it2) then
                        bondhist(it1,it2) = bondhist(it1,it2) + 1      
                    end if 
                end do
            end do  
            
        end do
    end do 
    
    
    
    !half 1-1,2-2 and 3-3 counts for double counting
    do i=1, typecount
        bondhist(i,i) = bondhist(i,i) / 2.0_dp 
    end do   
    
    
    
    !sum of bonds for normalization
    do i=1, typecount
        do j=i, typecount
            bondhist_sum = bondhist_sum + bondhist(i,j)        
        end do
    end do
    
    
    !fraction of bond types
    do i=1, typecount
        do j=i, typecount
            if(bondhist_sum/=0.0_dp) then
            bondhist_fraction(i,j) = bondhist(i,j) / bondhist_sum 
            else
                bondhist_fraction(i,j) = 0.0_dp
            end if
        end do
    end do        
    
    
    !Write out data
    if(headers_done==0) then
        write(25,110)
        110 format('BOND TYPE FRACTIONS (Fraction of i-j bonds from total number of bonds)') 
        write(25,120)
        120 format('      Frame',$)    
        do i=1, typecount
            do j=i, typecount
                write(25,130) i,j
                130     format(',',(6x,i2,1x,i2),$)     
            end do
        end do  
        write(25,135) 
        135 format(',      Bonds')   
        
    end if
    
    
    write(25,140) frame 
    140 format(i11,$)     
    do i=1, typecount
        do j=i, typecount
            write(25,150) bondhist_fraction(i,j)
            150     format(',',f11.6,$)  
            
            !         -Write out feature file 
            write(90,151) bondhist_fraction(i,j)
            151     format(',',f11.6,$)  
            
            
        end do
    end do 
    write(25,160) int(bondhist_sum)
    160 format(',',i11)
    
    !Write out feature file      
    write(90,161) int(bondhist_sum)
    161 format(',',i11,$)  
    
    
    
    
    
    
    DEALLOCATE(bondhist)
    DEALLOCATE(bondhist_fraction)
    
    return
end 



!END OF CAL_BTYPE ROUTINE
!=======================================================================      



!=======================================================================
!START OF CAL_GRSQ ROUTINE
!Partial and total radial distribution functions/total structure factor 

subroutine CAL_GRSQ
    
    use KINDS, only: dp
    use VARIABLES, only: c_lab, con_pi, frame, headers_done, in_delr,  &
    in_density, in_gr_points, in_sq_points, natoms, typecount,         &
    xc, yc, zc, zl, xl, yl, zl, xl2inv, yl2inv, zl2inv  
    IMPLICIT NONE

    
    !Local     
    integer :: i, j, k
    integer :: nnbin
    integer :: typeA, typeB
    integer :: nqpts, nqstart, nqfinish
    integer :: nnq, ir, isign, m, mm
    real(dp) :: xposi, yposi, zposi
    real(dp) :: xu, yu, zu, rdist, maxr
    real(dp) :: rho, prodel
    real(dp) :: delq, sqconst
    real(dp), allocatable :: g2_part(:,:,:)
    real(dp), allocatable :: g2_total(:)
    real(dp) :: sq_data(20000)  
    real(dp) :: sq_grtemp(10000)
    real(dp) :: sq_final(10000)
    
    
    !Number of point in the S(q) (must be 2^n+1)
    nqpts = in_sq_points
    
    nqstart = 1
    nqfinish = nqpts
    
    !Reduced density
    if(in_density==0.0_dp) then
        rho = natoms(frame)/(xl*yl*zl)
    end if
    if(in_density>0.0_dp) then
        rho = in_density
    end if
    
 
    !Maximum range of g(r) is half cell length 
    maxr = min(xl,yl,zl) * 0.5_dp   
    
    
    !Check that g(r) range isn't larger than half cell length      
    if(in_gr_points>int(maxr/in_delr)) then
        print*,'ERROR: in_gr_points > half minimum cell length'
        STOP
    end if
    
    !Factor for histogram to g(r)
    prodel = 2.0_dp*con_pi*in_delr*natoms(frame)*rho 
    
    !delq determination from delr
    delq = 2.0_dp*con_pi / (in_delr*(2.0_dp*nqpts-2.0_dp)) 
    
    !Factor for S(q)
    sqconst = 4.0_dp*con_pi*rho*(in_delr**2.0_dp)/delq      
    
    
    !Allocate array
    ALLOCATE(g2_part(typecount,typecount,in_gr_points))
    ALLOCATE(g2_total(in_gr_points))
    
    
    !Zero arrays
    g2_part = 0.0_dp            
    g2_total = 0.0_dp         
    
    
    
    !Calculate g(r) for frame
    do i=1, natoms(frame)
        
        typeA = c_lab(i)
        xposi = xc(i)
        yposi = yc(i)
        zposi = zc(i)
        
        do j=1, natoms(frame)
            
            if(i/=j) then
                
                typeB = c_lab(j)  
                xu = xc(j) - xposi
                yu = yc(j) - yposi
                zu = zc(j) - zposi
                
                !Periodic boundaries
                xu = xu - (int(xu*xl2inv)*xl)
                yu = yu - (int(yu*yl2inv)*yl)
                zu = zu - (int(zu*zl2inv)*zl)
                
                rdist = sqrt(xu*xu+yu*yu+zu*zu)
                
                
                !Separation histogram for g(r) calculation
                nnbin = rdist/in_delr + 1.5
                if(nnbin<=in_gr_points) then
                    g2_part(typeA,typeB,nnbin) = g2_part(typeA,typeB,nnbin)  + 1
                end if
            end if
            
        end do
    end do
    
    
    !Half diagonal totals (due to double count)
    do i=1, typecount
        do j=1, in_gr_points
            g2_part(i,i,j) = g2_part(i,i,j) / 2.0_dp
        end do
    end do
    
    
    
    !Histogram to partial g(r)
    do i=1, typecount      
        do j=1, typecount
            do k=2, in_gr_points           !start at 2 to avoid div 0 
                g2_part(i,j,k) = g2_part(i,j,k) / (prodel*((k-1)*in_delr)*((k-1)*in_delr))
            end do
        end do
    end do  
    
    !Total g(r) is sum of unique partials
    do i=1, in_gr_points  
        do j=1, typecount
            do k=j, typecount
                g2_total(i) = g2_total(i) + g2_part(j,k,i)
            end do
        end do
    end do  
    
    
    
    
    
    
    !S(q) calculation 
    !
    !All even elements (complex) set to zero
    !and all elements from nqpts to 2nqpts-2 to zero
    !nnq is the number of pairs of complex numbers
    !in the data array
    
    
    nnq = 2*nqpts-2
    
    if(mod(nnq,2)/=0) then
        print*,'ERROR: nnq = 2*nqpts-2 MUST bE EVEN'
        STOP
    endif      
    
    !Copy total gr to array large enough for fft, pad out with 1
    do i=1, 2*nqpts
        if(i<=in_gr_points) then
        sq_grtemp(i) = g2_total(i)
        else
            sq_grtemp(i) = 1.0_dp
        end if
    end do
    
    do i=1, 2*nqpts-1, 2
        ir = (i+1)/2
        sq_data(i)=(ir-1)*(sq_grtemp(ir)-1.0_dp)
    end do      
    
    !Zero sq_data from nqpts to 2*nqpts      
    do i=2*(nqpts+1)-1, 2*(2*nqpts-2)-1, 2
        sq_data(i)=0.0_dp
    end do      
    
    !Zero sq_data for all even i      
    do i = 2, 2*(2*nqpts-2), 2
        sq_data(i) = 0.0_dp
    end do   
    
    isign = 1      
    
    CALL CAL_GRSQ_FFT(sq_data,nnq,isign)      
    
    do i=2*(nqstart),2*(nqfinish),2
        m  = i/2 - 1
        mm = i/2 - nqstart + 1
        sq_final(mm) = 1.0_dp + sqconst*sq_data(i)/m
    end do      
    
    
    
    
    
    
    !Write out g(r)
    
    if(headers_done==0) then
        !Title (1st line)      
        write(51,10)
        10  format('RADIAL DISTRIBUTION FUNCTION g(r) - 1st line is r followed by total and partial g(r) blocks for each frame') 
        
        !Write label before r values
        write(51,15) 
        15  format('      Frame',',      Type1',',      Type2',$)
        
        !Write out r values  
        do i=1, in_gr_points
            write(51,20) (i-1)*in_delr
            20    format(',',f11.6,$)
        end do
        
        write(51,*) !new line
    end if
    
    !Write total gr label
    write(51,25) frame
    25  format(i11,',      Total',',      Total',$)      
    
    !Write total gr out   
    do i=1, in_gr_points
        write(51,30) g2_total(i)
        30    format(',',f11.6,$)      
    end do
    
    !Write out to featureset file
    write(90,31)
    31  format(',      Total',',      Total',$)    
    do i=1, in_gr_points
        write(90,32) g2_total(i)
        32    format(',',f11.6,$)
    end do
    
    
    write(51,*)                                                         !New line
    
    if(typecount>1) then                                                !Only output partials for more than 1 component
        do i=1, typecount
            do j=i, typecount

                write(51,35) frame,i,j           
                35       format(i11,',',(9x,i2),',',(9x,i2),$)   
                
                !Write out to featureset file
                write(90,36) i,j
                36       format(',',(9x,i2),',',(9x,i2),$)         
                
                do k=1, in_gr_points
                    write(51,40) g2_part(i,j,k)
                    40         format(',',f11.6,$)
                end do
                
                !Write out to featureset file     
                do k=1, in_gr_points
                    write(90,41) g2_part(i,j,k)
                    41         format(',',f11.6,$)
                end do     
                
                write(51,*)  !New line
            end do
        end do
    end if
    
    
    
    
    
    !Write out S(q)
    if(headers_done==0) then     
        write(55,50)
        50  format('STRUCTURE FACTOR S(q)  q(angstrom^-1) versus S(q)') 
        
        !Write label before r values
        write(55,60) 
        60  format('      Frame',$)
        
        !Write out q values  
        do i=1, nqpts
            write(55,70) (i-1)*delq
            70    format(',',f11.6,$)
        end do
        
        write(55,*) !New line
    end if
    
    !Write total sq label
    write(55,80) frame
    80  format(i11,$)      
    
    !Write total sq out   
    do i=1, nqpts
        write(55,90) sq_final(i)
        90    format(',',f11.6,$)      
    end do
    
    !Write out to featureset file
    do i=1, nqpts
        write(90,91) sq_final(i)
        91    format(',',f11.6,$)      
    end do      
    
    write(55,*) !New line
    
    
    
    
    
    !Deallocate arrays
    DEALLOCATE(g2_part)
    DEALLOCATE(g2_total)      
    
    return
end 


!END OF CAL_GRSQ ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_GRSQ_FFT ROUTINE

subroutine CAL_GRSQ_FFT(data,nnq,isign)
    
    use KINDS, only: dp
    IMPLICIT NONE
    
    
    integer I, J, M, N, MMAX, ISTEP, ISIGN, NNQ
    real(dp) :: tempr, tempi
    real(dp) :: WR,WI,WPR,WPI,WTEMP,THETA
    real(dp) :: data(2*NNQ)

    
    N=2*NNQ
    J=1
        
    DO 11 I=1,N,2
    IF(J>I) THEN
        TEMPR=DATA(J)
        TEMPI=DATA(J+1)
        DATA(J)=DATA(I)
        DATA(J+1)=DATA(I+1)
        DATA(I)=TEMPR
        DATA(I+1)=TEMPI
    ENDIF
    M=N/2
    1    IF((M>=2).AND.(J>M)) THEN
        J=J-M
        M=M/2
        GO TO 1
    ENDIF
    J=J+M
    11 CONTINUE
    
    !BIT REVERSAL COMPLETED
    MMAX=2
    2  IF  (N>MMAX)  THEN
        ISTEP=2*MMAX
        
        THETA=6.28318530717959D0/(ISIGN*MMAX)
        WPR=-2.D0*DSIN(0.5D0*THETA)**2
        WPI=DSIN(THETA)
        WR=1.D0
        WI=0.D0
        
        
        DO 13 M=1,MMAX,2
        
        DO 12 I=M,N,ISTEP
        J=I+MMAX
        TEMPR=SNGL(WR)*DATA(J)-SNGL(WI)*DATA(J+1)
        TEMPI=SNGL(WR)*DATA(J+1)+SNGL(WI)*DATA(J)
        DATA(J)=DATA(I)-TEMPR
        DATA(J+1)=DATA(I+1)-TEMPI
        DATA(I)=DATA(I)+TEMPR
        DATA(I+1)=DATA(I+1)+TEMPI
        
        
        12     CONTINUE
        WTEMP=WR
        WR=WR*WPR-WI*WPI+WR
        WI=WI*WPR+WTEMP*WPI+WI
        13   CONTINUE
        
        MMAX=ISTEP
        GO TO 2
    ENDIF
    
    RETURN
END


!END OF CAL_GRSQ_FFT ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_G3 ROUTINE
!Partial and total bond angle distribution

subroutine CAL_G3
    
    use KINDS, only: dp
    use VARIABLES, only: c_lab, con_pi, frame, headers_done,           &
    natoms, natoms_max, ncord, nnlist, typecount,                      &
    lc, xl, yl, zl, xl2inv, yl2inv, zl2inv, xc, yc, zc,                &
    in_xyz_prec, xyz_prec_str
    IMPLICIT NONE
    
    
    !Local 
    integer :: i, j, k, l
    integer :: it1, it2
    integer :: count1                                                   !total number of angles
    integer :: count2(typecount,typecount,typecount)                    !number of angles between A-B-C           
    integer :: count3(natoms_max)                                       !number of angles for each atom
    integer :: typeA, typeB, typeC
    integer :: phis                                                     !integer cast bond angle
    real(dp) :: binsize
    real(dp) :: tol 
    real(dp) :: xu1, yu1, zu1, rr1
    real(dp) :: xu2, yu2, zu2, rr2
    real(dp) :: adotb, magab, cosphi, coscheck, phi
    real(dp), allocatable :: g3_part(:,:,:,:)
    real(dp), allocatable :: g3_part_mean(:,:,:)    
    real(dp), allocatable :: g3_part_stdev(:,:,:)    
    real(dp) :: g3_total(180)
    real(dp) :: g3_total_mean    
    real(dp) :: g3_total_stdev   
    real(dp) :: g3_pavg(natoms_max)
    real(dp) :: g3_pmax(natoms_max)
    real(dp) :: g3_pmin(natoms_max)
    integer :: g3_prec
    character(len=50) :: g3_prec_str
    character(len=200) :: writeg3

    
    ALLOCATE(g3_part(typecount,typecount,typecount,180))
    ALLOCATE(g3_part_mean(typecount,typecount,typecount))
    ALLOCATE(g3_part_stdev(typecount,typecount,typecount))
    

    
    g3_part = 0.0_dp
    g3_part_mean = 0.0_dp
    g3_part_stdev = 0.0_dp
    g3_total = 0.0_dp
    g3_total_mean = 0.0_dp
    g3_total_stdev = 0.0_dp
    

    !Binsize 
    binsize = 1.0_dp  
    
    !Calculate bond angles    
    g3_pavg = 0.0_dp
    g3_pmax = 0.0_dp
    g3_pmin = 180.0_dp
    tol = 0.00001_dp
    count1 = 0
    count2 = 0
    count3 = 0
    do i=1, natoms(frame)
        typeA = c_lab(i)
        
        do j=1, ncord(i)
            it1 = nnlist(i,j)
            
            typeB = c_lab(it1)
            xu1 = xc(i) - xc(it1)
            yu1 = yc(i) - yc(it1)
            zu1 = zc(i) - zc(it1)
            
            xu1 = xu1 - (int(xu1*xl2inv)*xl)
            yu1 = yu1 - (int(yu1*yl2inv)*yl)
            zu1 = zu1 - (int(zu1*zl2inv)*zl)
            
            rr1=xu1*xu1+yu1*yu1+zu1*zu1
            
            do k=j+1, ncord(i)  !Does not take the neighbouring atoms of first shell atoms into account
                it2 = nnlist(i,k)
                
                typeC = c_lab(it2)
                xu2 = xc(i) - xc(it2)
                yu2 = yc(i) - yc(it2)
                zu2 = zc(i) - zc(it2)
                
                xu2 = xu2 - (int(xu2*xl2inv)*xl)
                yu2 = yu2 - (int(yu2*yl2inv)*yl)
                zu2 = zu2 - (int(zu2*zl2inv)*zl)
                
                rr2=xu2*xu2+yu2*yu2+zu2*zu2
                
                adotb = xu2*xu1+yu2*yu1+zu2*zu1
                magab = sqrt(rr2*rr1)
                cosphi  = adotb/magab
                
                coscheck = 1.0_dp + cosphi
                if(abs(coscheck)<=tol) cosphi = -0.99999_dp
                phi = acos(cosphi)
                phi=phi*180.0_dp/(con_pi)
                
                phis = int((phi+ (binsize*0.5_dp))/binsize) 
                if(phis>180) phis = 360-phis
                if(phis==0) phis = 1                                    !unrealistic small angles, avoid div0
                
                count1 = count1 + 1                                     !total angles
                g3_total(phis) = g3_total(phis) + 1
                
                count2(typeA, typeB, typeC) = count2(typeA, typeB, typeC) + 1
                g3_part(typeA,typeB,typeC,phis) = g3_part(typeA,typeB,typeC,phis) + 1.0_dp

                count3(i) = count3(i) + 1
                g3_pavg(i) = g3_pavg(i) + phis
                if (phis>g3_pmax(i)) then
                    g3_pmax(i) = phis
                end if
                if (phis<g3_pmin(i)) then
                    g3_pmin(i) = phis
                end if

            end do
        end do
        
        if(count3(i)>0) then
            g3_pavg(i) = g3_pavg(i) / count3(i)                !average bond length from particle i
        end if
    end do  
    
    !Write out xyz file
    g3_prec = 1
    write(g3_prec_str,*) g3_prec 
    writeg3 = '(a,2x,3(F10.'//xyz_prec_str//',2X),3(F10.'//g3_prec_str//',2X),i6)'
    write(56,'(i10,/)') natoms(frame)
    do i=1, natoms(frame)
        write(56,writeg3) lc(i),xc(i),yc(i),zc(i),g3_pavg(i),g3_pmax(i),g3_pmin(i),count3(i)
    end do  

    !Add off-diagonals because in g3, 1-2 = 2-1  
    do i=1, typecount
        do j=1, typecount
            do k=j+1, typecount
                do l=1, 180           
                    g3_part(i,j,k,l) = g3_part(i,j,k,l) + g3_part(i,k,j,l)
                end do
            end do
        end do
    end do
    
    !Normalize distributions (summed off-diagonal and diagonals)
    do i=1, typecount
        do j=1, typecount
            do k=j, typecount
                do l=1, 180
                    if(count1>0) then
                        g3_part(i,j,k,l) = g3_part(i,j,k,l) / (count1*binsize)
                    else
                        g3_part(i,j,k,l) = 0.0_dp
                    end if
                end do
            end do
        end do  
    end do
    do i=1, 180
        if(count1>0) then
            g3_total(i) = g3_total(i) / (count1*binsize)
        else
            g3_total(i) = 0.0_dp
        end if
    end do
    
    
    
    !Mean of distribution (sum of P(i)*i)
    do i=1, typecount
        do j=1, typecount
            do k=1, typecount
                do l=1, 180
                    if(count2(i,j,k)>0) then
                        g3_part_mean(i,j,k) = g3_part_mean(i,j,k) +    &!g3_part_mean is NOT normalized (g3 partials added up to g3 total)
                        g3_part(i,j,k,l) * count1 * real(l,dp) /       &!thus to work out mean & std dev, need to normalize by g3 part angles  
                        count2(i,j,k)                                   !not total angles  
                    else                                                
                        g3_part_mean(i,j,k) = 0.0_dp
                    end if
                end do
            end do
        end do
    end do
    do i=1, 180
        g3_total_mean = g3_total_mean + g3_total(i) * i
    end do
    
    
    !Standard deviation (sqrt(sum of((i-mean)^2*P(i))))
    do i=1, typecount
        do j=1, typecount
            do k=1, typecount
                do l=1, 180
                    if(count2(i,j,k)>0) then
                        g3_part_stdev(i,j,k) = g3_part_stdev(i,j,k) +  &!Similar to the mean of partial g3s, needs to be re-normalized
                        (g3_part(i,j,k,l)*count1/count2(i,j,k))        &
                        *(real(l,dp)-g3_part_mean(i,j,k))**2 
                    else
                        g3_part_stdev(i,j,k) = 0.0_dp
                    end if
                end do
                
                g3_part_stdev(i,j,k) = sqrt(g3_part_stdev(i,j,k))
                
            end do
        end do
    end do
    do i=1, 180
      g3_total_stdev = g3_total_stdev + g3_total(i)*(real(i,dp)-g3_total_mean)**2 
    end do
    g3_total_stdev = sqrt(g3_total_stdev) 


    !Write out g3(theta)
    
    if(headers_done==0) then
        !Title (1st line)      
        write(52,10)
        10  format('ANGULAR DISTRIBUTION - blocks for & 
        each frame with 1st line being total g3(theta) beginning with frame number') 
        
        !Write label before angle values
        write(52,15) 
        15  format('      Frame',',      Type1',',      Type2',',      Type3', &
                   ',     Angles',',        Avg',',    Std Dev',$)
        
        !Write out angle values  
        do i=1, 180
            write(52,20) i
            20    format(',',i11,$)
        end do
        
        write(52,*) !New line
    end if
    
    !Write total angle label
    write(52,25) frame,count1,g3_total_mean,g3_total_stdev
    25  format(i11,',      Total',',      Total',',      Total',',',i11,',',f11.1,',',f11.1,$)      
    
    !Write total g3 out   
    do i=1, 180
        write(52,30) g3_total(i)
        30  format(',',f11.1,$)      
    end do
    
    !Write out to featureset file
    write(90,31) g3_total_mean,g3_total_stdev
    31  format(',',f11.1,',',f11.1,$)
    do i=1, 180
        write(90,32) g3_total(i)
        32  format(',',f11.1,$)      
    end do         
    
    
    write(52,*) !New line
    
    if(typecount>1) then     !Only output partials for more than 1 component  
        !Write partial g3 out  
        do i=1, typecount
            do j=1, typecount
                do k=j, typecount
                    !Write partial g3 labels 
                    write(52,35) frame,i,j,k,count2(i,j,k),g3_part_mean(i,j,k), g3_part_stdev(i,j,k)        
                    35  format(i11,(',',9x,i2),(',',9x,i2),(',',9x,i2),',',i11,',',f11.6,',',f11.6,$)     
                    
                    do l=1, 180
                        write(52,40) g3_part(i,j,k,l)
                        40  format(',',f11.6,$)
                    end do
                    write(52,*)  !New line
                end do
            end do
        end do   
    end if

    
    DEALLOCATE(g3_part)
    DEALLOCATE(g3_part_mean)
    DEALLOCATE(g3_part_stdev)


    return
end 


!END OF CAL_G3 ROUTINE
!=======================================================================


!=======================================================================
!START OF CAL_BANGLE ROUTINE
!Partial and total 2nd order bond angle distribution (includes 2nd NN shell)

subroutine CAL_BANGLE
    
    use KINDS, only: dp
    use VARIABLES, only: c_lab, con_pi, frame, headers_done,           &
    natoms, natoms_max, ncord, nnlist, typecount,                      &
    lc, xl, yl, zl, xl2inv, yl2inv, zl2inv, xc, yc, zc,                &
    in_xyz_prec, xyz_prec_str
    IMPLICIT NONE
    
    
    !Local 
    integer :: i, j, k, l
    integer :: it1, it2
    integer :: count1                                                   !total number of angles
    integer :: count2(typecount,typecount,typecount)                    !number of angles between A-B-C           
    integer :: count3(natoms_max)                                       !number of angles for each atom
    integer :: typeA, typeB, typeC
    integer :: phis                                                     !integer cast bond angle
    real(dp) :: binsize
    real(dp) :: tol 
    real(dp) :: xu1, yu1, zu1, rr1
    real(dp) :: xu2, yu2, zu2, rr2
    real(dp) :: adotb, magab, cosphi, coscheck1, coscheck2, phi
    real(dp), allocatable :: bangle_part(:,:,:,:)
    real(dp), allocatable :: bangle_part_mean(:,:,:)    
    real(dp), allocatable :: bangle_part_stdev(:,:,:)    
    real(dp) :: bangle_total(180)
    real(dp) :: bangle_total_mean    
    real(dp) :: bangle_total_stdev   
    real(dp) :: bangle_pavg(natoms_max)
    real(dp) :: bangle_pmax(natoms_max)
    real(dp) :: bangle_pmin(natoms_max)
    integer :: bangle_prec
    character(len=50) :: bangle_prec_str
    character(len=200) :: writebangle

    
    ALLOCATE(bangle_part(typecount,typecount,typecount,180))
    ALLOCATE(bangle_part_mean(typecount,typecount,typecount))
    ALLOCATE(bangle_part_stdev(typecount,typecount,typecount))
    

    bangle_part = 0.0_dp
    bangle_part_mean = 0.0_dp
    bangle_part_stdev = 0.0_dp
    bangle_total = 0.0_dp
    bangle_total_mean = 0.0_dp
    bangle_total_stdev = 0.0_dp
    
    !Binsize 
    binsize = 1.0_dp  
    
    !Calculate bond angles    
    bangle_pavg = 0.0_dp
    bangle_pmax = 0.0_dp
    bangle_pmin = 180.0_dp
    tol = 0.00001_dp
    count1 = 0
    count2 = 0
    count3 = 0
    do i=1, natoms(frame)
        typeA = c_lab(i)
        
        do j=1, ncord(i)
            it1 = nnlist(i,j)
            
            typeB = c_lab(it1)
            xu1 = xc(i) - xc(it1)
            yu1 = yc(i) - yc(it1)
            zu1 = zc(i) - zc(it1)
            
            xu1 = xu1 - (int(xu1*xl2inv)*xl)
            yu1 = yu1 - (int(yu1*yl2inv)*yl)
            zu1 = zu1 - (int(zu1*zl2inv)*zl)
            
            rr1=xu1*xu1+yu1*yu1+zu1*zu1
            
            do k=1, ncord(it1)
                it2 = nnlist(it1,k)
                if(it2.eq.i) cycle  

                typeC = c_lab(it2)
                xu2 = xc(i) - xc(it2)
                yu2 = yc(i) - yc(it2)
                zu2 = zc(i) - zc(it2)
                
                xu2 = xu2 - (int(xu2*xl2inv)*xl)
                yu2 = yu2 - (int(yu2*yl2inv)*yl)
                zu2 = zu2 - (int(zu2*zl2inv)*zl)
                
                rr2=xu2*xu2+yu2*yu2+zu2*zu2
                
                adotb = xu2*xu1+yu2*yu1+zu2*zu1
                magab = sqrt(rr2*rr1)
                cosphi  = adotb/magab
                
                coscheck1 = cosphi + 1.0_dp
                if(abs(coscheck1)<=tol) cosphi = -0.99999_dp
                coscheck2 = cosphi - 1.0_dp
                if(abs(coscheck2)<=tol) cosphi = 1.0_dp
                phi = acos(cosphi)
                phi=phi*180.0_dp/(con_pi)
                
                phis = int((phi+ (binsize*0.5_dp))/binsize) 
                if(phis>180) phis = 360-phis
                if(phis==0) phis = 1                                    !unrealistic small angles, avoid div0
                
                count1 = count1 + 1                                     !total angles
                bangle_total(phis) = bangle_total(phis) + 1
                
                count2(typeA, typeB, typeC) = count2(typeA, typeB, typeC) + 1
                bangle_part(typeA,typeB,typeC,phis) = bangle_part(typeA,typeB,typeC,phis) + 1.0_dp

                count3(i) = count3(i) + 1
                bangle_pavg(i) = bangle_pavg(i) + phis
                if (phis>bangle_pmax(i)) then
                    bangle_pmax(i) = phis
                end if
                if (phis<bangle_pmin(i)) then
                    bangle_pmin(i) = phis
                end if

            end do
        end do
 
        if(count3(i)>0) then
            bangle_pavg(i) = bangle_pavg(i) / count3(i)                !average bond length from particle i
        end if
    end do  
    
    !Write out xyz file
    bangle_prec = 1
    write(bangle_prec_str,*) bangle_prec 
    writebangle = '(a,2x,3(F10.'//xyz_prec_str//',2X),3(F10.'//bangle_prec_str//',2X),i6)'
    write(57,'(i10,/)') natoms(frame)        
    do i=1, natoms(frame)
        write(57,writebangle) lc(i),xc(i),yc(i),zc(i),bangle_pavg(i),bangle_pmax(i),bangle_pmin(i),count3(i)
    end do  

    !Add off-diagonals because in bond angles, 1-2 = 2-1  
    do i=1, typecount
        do j=1, typecount
            do k=j+1, typecount
                do l=1, 180           
                    bangle_part(i,j,k,l) = bangle_part(i,j,k,l) + bangle_part(i,k,j,l)
                end do
            end do
        end do
    end do
    
    !Normalize distributions (summed off-diagonal and diagonals)
    do i=1, typecount
        do j=1, typecount
            do k=j, typecount
                do l=1, 180
                    if(count1>0) then
                        bangle_part(i,j,k,l) = bangle_part(i,j,k,l) / (count1*binsize)
                    else
                        bangle_part(i,j,k,l) = 0.0_dp
                    end if
                end do
            end do
        end do  
    end do
    do i=1, 180
        if(count1>0) then
            bangle_total(i) = bangle_total(i) / (count1*binsize)
        else
            bangle_total(i) = 0.0_dp
        end if
    end do
    
    
    
    !Mean of distribution (sum of P(i)*i)
    do i=1, typecount
        do j=1, typecount
            do k=1, typecount
                do l=1, 180
                    if(count2(i,j,k)>0) then
                        bangle_part_mean(i,j,k) = bangle_part_mean(i,j,k) +    &!bangle_part_mean is NOT normalized (bangle partials added up to bangle total)
                        bangle_part(i,j,k,l) * count1 * real(l,dp) /       &!thus to work out mean & std dev, need to normalize by bangle part angles  
                        count2(i,j,k)                                   !not total angles  
                    else                                                
                        bangle_part_mean(i,j,k) = 0.0_dp
                    end if
                end do
            end do
        end do
    end do
    do i=1, 180
        bangle_total_mean = bangle_total_mean + bangle_total(i) * i
    end do
    
    
    !Standard deviation (sqrt(sum of((i-mean)^2*P(i))))
    do i=1, typecount
        do j=1, typecount
            do k=1, typecount
                do l=1, 180
                    if(count2(i,j,k)>0) then
                        bangle_part_stdev(i,j,k) = bangle_part_stdev(i,j,k) +  &!Similar to the mean of partial bangles, needs to be re-normalized
                        (bangle_part(i,j,k,l)*count1/count2(i,j,k))        &
                        *(real(l,dp)-bangle_part_mean(i,j,k))**2 
                    else
                        bangle_part_stdev(i,j,k) = 0.0_dp
                    end if
                end do
                
                bangle_part_stdev(i,j,k) = sqrt(bangle_part_stdev(i,j,k))
                
            end do
        end do
    end do
    do i=1, 180
      bangle_total_stdev = bangle_total_stdev + bangle_total(i)*(real(i,dp)-bangle_total_mean)**2 
    end do
    bangle_total_stdev = sqrt(bangle_total_stdev) 


    !Write out bangle(theta)
    
    if(headers_done==0) then
        !Title (1st line)      
        write(53,10)
        10  format('SECOND ORDER ANGULAR DISTRIBUTION - blocks for & 
        each frame with 1st line being total bangle(theta) beginning with frame number') 
        
        !Write label before angle values
        write(53,15) 
        15  format('      Frame',',      Type1',',      Type2',',      Type3', &
                   ',     Angles',',        Avg',',    Std Dev',$)
        
        !Write out angle values  
        do i=1, 180
            write(53,20) i
            20    format(',',i11,$)
        end do
        
        write(53,*) !New line
    end if
    
    !Write total angle label
    write(53,25) frame,count1,bangle_total_mean,bangle_total_stdev
    25  format(i11,',      Total',',      Total',',      Total',',',i11,',',f11.1,',',f11.1,$)      
    
    !Write total bangle out   
    do i=1, 180
        write(53,30) bangle_total(i)
        30  format(',',f11.1,$)      
    end do
    
    !Write out to featureset file
    write(90,31) bangle_total_mean,bangle_total_stdev
    31  format(',',f11.1,',',f11.1,$)
    do i=1, 180
        write(90,32) bangle_total(i)
        32  format(',',f11.1,$)      
    end do         
    
    
    write(53,*) !New line
    
    if(typecount>1) then     !Only output partials for more than 1 component  
        !Write partial bangle out  
        do i=1, typecount
            do j=1, typecount
                do k=j, typecount
                    !Write partial bangle labels 
                    write(53,35) frame,i,j,k,count2(i,j,k),bangle_part_mean(i,j,k), bangle_part_stdev(i,j,k)        
                    35  format(i11,(',',9x,i2),(',',9x,i2),(',',9x,i2),',',i11,',',f11.6,',',f11.6,$)     
                    
                    do l=1, 180
                        write(53,40) bangle_part(i,j,k,l)
                        40  format(',',f11.6,$)
                    end do
                    write(53,*)  !New line
                end do
            end do
        end do   
    end if

    
    DEALLOCATE(bangle_part)
    DEALLOCATE(bangle_part_mean)
    DEALLOCATE(bangle_part_stdev)


    return
end 


!END OF CAL_BANGLE ROUTINE
!=======================================================================


!=======================================================================
!START OF CAL_BTORSION ROUTINE
!Bond torsion statistics

subroutine CAL_BTORSION
    
    use KINDS, only: dp
    use VARIABLES, only: c_lab, con_pi, frame, headers_done,           &
    natoms, natoms_max, ncord, nnlist, typecount,                      &
    lc, xl, yl, zl, xl2inv, yl2inv, zl2inv, xc, yc, zc,                &
    in_xyz_prec, xyz_prec_str
    IMPLICIT NONE
    
    
    !Local 
    integer :: i, j, k, l, m
    integer :: it1, it2, it3
    integer :: btors_neg_cnt(natoms_max)                                !number of torsion angles for each atom
    integer :: btors_pos_cnt(natoms_max)                               
    integer :: count1neg                                                !total number of angles
    integer :: count2neg(typecount,typecount,typecount,typecount)       !number of angles between A-B-C-D
    integer :: count1pos                                               
    integer :: count2pos(typecount,typecount,typecount,typecount)      
    integer :: typeA, typeB, typeC, typeD
    integer :: phis                                                     !integer cast bond angle
    real(dp) :: binsize
    real(dp) :: tol,tolvec 
    real(dp) :: xu1, yu1, zu1
    real(dp) :: xu2, yu2, zu2
    real(dp) :: xu3, yu3, zu3
    real(dp) :: mag1, mag2, mag3
    real(dp) :: xm1, ym1, zm1
    real(dp) :: n1dotn2, m1dotn2
    real(dp) :: xn1, yn1, zn1, magn1, rr1
    real(dp) :: xn2, yn2, zn2, magn2, rr2
    real(dp) :: adotb, magab, cosphi, coscheck, phi
    real(dp), allocatable :: btors_neg_part(:,:,:,:,:)
    real(dp), allocatable :: btors_neg_part_mean(:,:,:,:)
    real(dp), allocatable :: btors_neg_part_stdev(:,:,:,:)
    real(dp), allocatable :: btors_pos_part(:,:,:,:,:)
    real(dp), allocatable :: btors_pos_part_mean(:,:,:,:)
    real(dp), allocatable :: btors_pos_part_stdev(:,:,:,:)
    real(dp) :: btors_neg_total(181)
    real(dp) :: btors_neg_total_mean
    real(dp) :: btors_neg_total_stdev
    real(dp) :: btors_pos_total(181)
    real(dp) :: btors_pos_total_mean
    real(dp) :: btors_pos_total_stdev
    real(dp) :: btors_neg_pavg(natoms_max)
    real(dp) :: btors_neg_pmax(natoms_max)
    real(dp) :: btors_neg_pmin(natoms_max)
    real(dp) :: btors_pos_pavg(natoms_max)
    real(dp) :: btors_pos_pmax(natoms_max)
    real(dp) :: btors_pos_pmin(natoms_max)
    integer :: btors_prec
    character(len=50) :: btors_prec_str
    character(len=200) :: writebtors

    ALLOCATE(btors_neg_part(typecount,typecount,typecount,typecount,181))
    ALLOCATE(btors_neg_part_mean(typecount,typecount,typecount,typecount))
    ALLOCATE(btors_neg_part_stdev(typecount,typecount,typecount,typecount))
    ALLOCATE(btors_pos_part(typecount,typecount,typecount,typecount,181))
    ALLOCATE(btors_pos_part_mean(typecount,typecount,typecount,typecount))
    ALLOCATE(btors_pos_part_stdev(typecount,typecount,typecount,typecount))

    btors_neg_part = 0.0_dp
    btors_neg_part_mean = 0.0_dp
    btors_neg_part_stdev = 0.0_dp
    btors_pos_part = 0.0_dp
    btors_pos_part_mean = 0.0_dp
    btors_pos_part_stdev = 0.0_dp
    btors_neg_total = 0.0_dp
    btors_neg_total_mean = 0.0_dp
    btors_neg_total_stdev = 0.0_dp
    btors_pos_total = 0.0_dp
    btors_pos_total_mean = 0.0_dp
    btors_pos_total_stdev = 0.0_dp

    !Binsize 
    binsize = 1.0_dp  
    
    !Calculate bond torsions from the normal unit vectors of 2 planes
    btors_neg_pavg = 0.0_dp
    btors_neg_pmax = -180.0_dp
    btors_neg_pmin = 0.0_dp
    btors_pos_pavg = 0.0_dp
    btors_pos_pmax = 0.0_dp
    btors_pos_pmin = 180.0_dp
    tol = 0.0001_dp
    tolvec = 0.1_dp
    btors_neg_cnt = 0
    btors_pos_cnt = 0
    count1neg = 0
    count2neg = 0
    count1pos = 0
    count2pos = 0
    do i=1, natoms(frame)
        typeA = c_lab(i)
 
        do j=1, ncord(i)
            it1 = nnlist(i,j)
            typeB = c_lab(it1)
            
            !Unit vector from atom i to atom it1
            xu1 = xc(it1) - xc(i)
            yu1 = yc(it1) - yc(i)
            zu1 = zc(it1) - zc(i)
            !xu1 = xu1 - (int(xu1*xl2inv)*xl)
            !yu1 = yu1 - (int(yu1*yl2inv)*yl)
            !zu1 = zu1 - (int(zu1*zl2inv)*zl)
            mag1 = sqrt(xu1*xu1 + yu1*yu1 + zu1*zu1)
            xu1 = xu1 / mag1
            yu1 = yu1 / mag1
            zu1 = zu1 / mag1
            
            do k=1, ncord(it1)
                it2 = nnlist(it1,k)
                if(it2.eq.i) cycle
                typeC = c_lab(it2)
                
                !Unit vector from atom it1 to atom it2
                xu2 = xc(it2) - xc(it1)
                yu2 = yc(it2) - yc(it1)
                zu2 = zc(it2) - zc(it1)
                !xu2 = xu2 - (int(xu2*xl2inv)*xl)
                !yu2 = yu2 - (int(yu2*yl2inv)*yl)
                !zu2 = zu2 - (int(zu2*zl2inv)*zl)
                mag2 = sqrt(xu2*xu2 + yu2*yu2 + zu2*zu2)
                xu2 = xu2 / mag2
                yu2 = yu2 / mag2
                zu2 = zu2 / mag2
                
                !Calculate normal unit vector perpendicular to plane 1 using cross products
                xn1 = yu1*zu2-zu1*yu2
                yn1 = zu1*xu2-xu1*zu2
                zn1 = xu1*yu2-yu1*xu2

                !Exclude combinations that couldn't form a plane
                if(abs(xn1)<tolvec.and.abs(yn1)<=tolvec.and.abs(zn1)<=tolvec) then
                    cycle
                end if
                
                !Calculate the last vector that forms an orthonormal frame with n1 and u2 (https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates)
                xm1 = yn1*zu2-zn1*yu2
                ym1 = zn1*xu2-xn1*zu2
                zm1 = xn1*yu2-yn1*xu2
                
                !magn1 = sqrt(xn1*xn1+yn1*yn1+zn1*zn1)
                !xn1 = xn1 / magn1
                !yn1 = yn1 / magn1
                !zn1 = zn1 / magn1
                !rr1=xn1*xn1+yn1*yn1+zn1*zn1
                
                do l=1, ncord(it2)
                    it3 = nnlist(it2,l)
                    if(it3.eq.i) cycle
                    if(it3.eq.it1) cycle
                    typeD = c_lab(it3)

                    !Unit vector from atom it2 to atom it3
                    xu3 = xc(it3) - xc(it2)
                    yu3 = yc(it3) - yc(it2)
                    zu3 = zc(it3) - zc(it2)
                    !xu3 = xu3 - (int(xu3*xl2inv)*xl)
                    !yu3 = yu3 - (int(yu3*yl2inv)*yl)
                    !zu3 = zu3 - (int(zu3*zl2inv)*zl)
                    mag3 = sqrt(xu3*xu3 + yu3*yu3 + zu3*zu3)
                    xu3 = xu3 / mag3
                    yu3 = yu3 / mag3
                    zu3 = zu3 / mag3

                    !Calculate normal unit vector perpendicular to plane 2 using cross products
                    xn2 = yu2*zu3-zu2*yu3
                    yn2 = zu2*xu3-xu2*zu3
                    zn2 = xu2*yu3-yu2*xu3

                    !Exclude combinations that couldn't form a plane
                    if(abs(xn2)<tolvec.and.abs(yn2)<=tolvec.and.abs(zn2)<=tolvec) then
                        cycle
                    end if

                    !magn2 = sqrt(xn2*xn2+yn2*yn2+zn2*zn2)
                    !xn2 = xn2 / magn2
                    !yn2 = yn2 / magn2
                    !zn2 = zn2 / magn2
                    !rr2=xn2*xn2+yn2*yn2+zn2*zn2

                    !Calculate angle between the normal unit vectors
                    n1dotn2 = xn1*xn2+yn1*yn2+zn1*zn2
                    m1dotn2 = xm1*xn2+ym1*yn2+zm1*zn2
                    phi = atan2(m1dotn2, n1dotn2) * 180 / con_pi
                    !magab = sqrt(rr2*rr1)
                    !cosphi  = adotb/magab

                    !coscheck = 1.0_dp + cosphi
                    !if(abs(coscheck)<=tol) cosphi = -0.99999_dp
                    !if(cosphi>=1.0_dp) cosphi = 0.99999_dp
                    !phi = acos(cosphi)
                    !phi = phi*180.0_dp/(con_pi)
                    
                    !if(phi<0) phi=-phi  !Comment out if signed torsion angle needed (causes atoms on opposite sides of a plane to differ)
                    if(phi<0) then
                        phis = int((phi- (binsize*0.5_dp))/binsize) 
                    else
                        phis = int((phi+ (binsize*0.5_dp))/binsize) 
                    end if
                    if(abs(phis)==0.or.abs(phis)==180) then
                        phis = abs(phis)
                    end if
                    
                    if(phis==0.or.phis==180) then
                        count1neg = count1neg + 1
                        count1pos = count1pos + 1
                        btors_neg_total(phis+1) = btors_neg_total(phis+1) + 1
                        btors_pos_total(phis+1) = btors_pos_total(phis+1) + 1
                        count2neg(typeA, typeB, typeC, typeD) = count2neg(typeA, typeB, typeC, typeD) + 1
                        count2pos(typeA, typeB, typeC, typeD) = count2pos(typeA, typeB, typeC, typeD) + 1
                        btors_neg_part(typeA,typeB,typeC,typeD,phis+1) = btors_neg_part(typeA,typeB,typeC,typeD,phis+1) + 1.0_dp
                        btors_pos_part(typeA,typeB,typeC,typeD,phis+1) = btors_pos_part(typeA,typeB,typeC,typeD,phis+1) + 1.0_dp
                        btors_neg_cnt(i) = btors_neg_cnt(i) + 1
                        btors_pos_cnt(i) = btors_pos_cnt(i) + 1
                        btors_neg_pavg(i) = btors_neg_pavg(i) - phis
                        btors_pos_pavg(i) = btors_pos_pavg(i) + phis
                        if (-phis>btors_neg_pmax(i)) btors_neg_pmax(i) = -phis
                        if (phis>btors_pos_pmax(i)) btors_pos_pmax(i) = phis
                        if (-phis<btors_neg_pmin(i)) btors_neg_pmin(i) = -phis
                        if (phis<btors_pos_pmin(i)) btors_pos_pmin(i) = phis
                    else if(phis<0) then
                        count1neg = count1neg + 1
                        btors_neg_total(-phis+1) = btors_neg_total(-phis+1) + 1
                        count2neg(typeA, typeB, typeC, typeD) = count2neg(typeA, typeB, typeC, typeD) + 1
                        btors_neg_part(typeA,typeB,typeC,typeD,-phis+1) = btors_neg_part(typeA,typeB,typeC,typeD,-phis+1) + 1.0_dp
                        btors_neg_cnt(i) = btors_neg_cnt(i) + 1
                        btors_neg_pavg(i) = btors_neg_pavg(i) + phis
                        if (phis>btors_neg_pmax(i)) btors_neg_pmax(i) = phis
                        if (phis<btors_neg_pmin(i)) btors_neg_pmin(i) = phis
                    else
                        count1pos = count1pos + 1
                        btors_pos_total(phis+1) = btors_pos_total(phis+1) + 1
                        count2pos(typeA, typeB, typeC, typeD) = count2pos(typeA, typeB, typeC, typeD) + 1
                        btors_pos_part(typeA,typeB,typeC,typeD,phis+1) = btors_pos_part(typeA,typeB,typeC,typeD,phis+1) + 1.0_dp
                        btors_pos_cnt(i) = btors_pos_cnt(i) + 1
                        btors_pos_pavg(i) = btors_pos_pavg(i) + phis
                        if (phis>btors_pos_pmax(i)) btors_pos_pmax(i) = phis
                        if (phis<btors_pos_pmin(i)) btors_pos_pmin(i) = phis
                    end if

                end do
            end do
        end do
        if(btors_neg_cnt(i)>0) btors_neg_pavg(i) = btors_neg_pavg(i) / btors_neg_cnt(i)
        if(btors_pos_cnt(i)>0) btors_pos_pavg(i) = btors_pos_pavg(i) / btors_pos_cnt(i)
    end do  
    
    !Write out xyz file
    btors_prec = 1
    write(btors_prec_str,*) btors_prec 
    writebtors = '(a,2x,3(F10.'//xyz_prec_str//',2X),6(F10.'//btors_prec_str//',2X),i6,i6)'
    write(58,'(i10,/)') natoms(frame)        
    do i=1, natoms(frame)
        write(58,writebtors) lc(i),xc(i),yc(i),zc(i),btors_pos_pavg(i),btors_neg_pavg(i),btors_neg_pmax(i),btors_neg_pmin(i)&
                ,btors_pos_pmax(i),btors_pos_pmin(i),btors_pos_cnt(i),btors_neg_cnt(i)
    end do  

    !Normalize distributions (summed off-diagonal and diagonals)
    do i=1, typecount
        do j=1, typecount
            do k=1, typecount
                do l=k, typecount  ! from k?
                    do m=1, 181
                        if(count1neg>0) then
                            btors_neg_part(i,j,k,l,m) = btors_neg_part(i,j,k,l,m) / (count1neg*binsize)
                        else
                            btors_neg_part(i,j,k,l,m) = 0.0_dp
                        end if
                        if(count1pos>0) then
                            btors_pos_part(i,j,k,l,m) = btors_pos_part(i,j,k,l,m) / (count1pos*binsize)
                        else
                            btors_pos_part(i,j,k,l,m) = 0.0_dp
                        end if
                    end do
                end do
            end do
        end do
    end do
    do i=1, 181  ! 0 to 180?
        if(count1neg>0) then
            btors_neg_total(i) = btors_neg_total(i) / (count1neg*binsize)
        else
            btors_neg_total(i) = 0.0_dp
        end if
        if(count1pos>0) then
            btors_pos_total(i) = btors_pos_total(i) / (count1pos*binsize)
        else
            btors_pos_total(i) = 0.0_dp
        end if
    end do

    !Mean of distribution (sum of P(i)*i)
    do i=1, typecount
        do j=1, typecount
            do k=1, typecount
                do l=1, typecount
                    do m=1, 181
                        if(count2neg(i,j,k,l)>0) then
                            btors_neg_part_mean(i,j,k,l) = btors_neg_part_mean(i,j,k,l) + &
                            btors_neg_part(i,j,k,l,m) * count1neg * real(m,dp) / count2neg(i,j,k,l) !normalize by part angles
                        else
                            btors_neg_part_mean(i,j,k,l) = 0.0_dp
                        end if
                        if(count2pos(i,j,k,l)>0) then
                            btors_pos_part_mean(i,j,k,l) = btors_pos_part_mean(i,j,k,l) + &
                            btors_pos_part(i,j,k,l,m) * count1pos * real(m,dp) / count2pos(i,j,k,l) !normalize by part angles
                        else
                            btors_pos_part_mean(i,j,k,l) = 0.0_dp
                        end if
                    end do
                end do
            end do
        end do
    end do
    do i=1, 181
        btors_neg_total_mean = btors_neg_total_mean + btors_neg_total(i) * i
        btors_pos_total_mean = btors_pos_total_mean + btors_pos_total(i) * i
    end do

    !Standard deviation (sqrt(sum of((i-mean)^2*P(i))))
    do i=1, typecount
        do j=1, typecount
            do k=1, typecount
                do l=1, typecount
                    do m=1, 181
                        if(count2neg(i,j,k,l)>0) then
                            btors_neg_part_stdev(i,j,k,l) = btors_neg_part_stdev(i,j,k,l) + &
                            (btors_neg_part(i,j,k,l,m)*count1neg/count2neg(i,j,k,l)) * (real(m,dp)-btors_neg_part_mean(i,j,k,l))**2
                        else
                            btors_neg_part_stdev(i,j,k,l) = 0.0_dp
                        end if
                        if(count2pos(i,j,k,l)>0) then
                            btors_pos_part_stdev(i,j,k,l) = btors_pos_part_stdev(i,j,k,l) + &
                            (btors_pos_part(i,j,k,l,m)*count1pos/count2pos(i,j,k,l)) * (real(m,dp)-btors_pos_part_mean(i,j,k,l))**2
                        else
                            btors_pos_part_stdev(i,j,k,l) = 0.0_dp
                        end if
                    end do
                    btors_neg_part_stdev(i,j,k,l) = sqrt(btors_neg_part_stdev(i,j,k,l))
                    btors_pos_part_stdev(i,j,k,l) = sqrt(btors_pos_part_stdev(i,j,k,l))
                end do
            end do
        end do
    end do
    do i=1, 181
        btors_neg_total_stdev = btors_neg_total_stdev + btors_neg_total(i)*(real(i,dp)-btors_neg_total_mean)**2
        btors_pos_total_stdev = btors_pos_total_stdev + btors_pos_total(i)*(real(i,dp)-btors_pos_total_mean)**2
    end do
    btors_neg_total_stdev = sqrt(btors_neg_total_stdev)
    btors_pos_total_stdev = sqrt(btors_pos_total_stdev)

    !Write out bond torsion
    if(headers_done==0) then
        !Title (1st line)
        write(54,10)
        10  format('TORSION DISTRIBUTION - blocks for &
        each frame with 1st line being total torison angle beginning with frame number')

        !Write label before torsion angle values
        write(54,15)
        15  format('      Frame',',      Type1',',      Type2',',      Type3',  &
                   ',      Type4',', Neg Angles',',    Neg Avg',',Neg Std Dev', &
                   ', Pos Angles',',    Pos Avg',',Pos Std Dev'$)

        !Write out torsion angle values
        do i=-180, 0
            write(54,20) i
            20    format(',',i11,$)
        end do
        do i=0, 180
            write(54,21) i
            21    format(',',i11,$)
        end do

        write(54,*) !New line
    end if

    !Write total torsion angle label
    write(54,25) frame,count1neg,btors_neg_total_mean,btors_neg_total_stdev,count1pos,btors_pos_total_mean,btors_pos_total_stdev
    25  format(i11,',      Total',',      Total',',      Total',',      Total',2(',',i11,',',f11.6,',',f11.6)$)

    !Write total bond torsion out
    do i=1, 181
        write(54,30) btors_neg_total(i),btors_pos_total(i)
        30  format(',',f11.6,',',f11.6,$)
    end do

    !Write out to featureset file
    write(90,31) btors_neg_total_mean,btors_neg_total_stdev,btors_pos_total_mean,btors_pos_total_stdev
    31  format(',',f11.6,',',f11.6,',',f11.6,',',f11.6,$)
    do i=1, 181
        write(90,32) btors_neg_total(i),btors_pos_total(i)
        32  format(',',f11.6,',',f11.6,$)
    end do

    write(54,*) !New line

    if(typecount>1) then     !Only output partials for more than 1 component
        !Write partial bond torsion out
        do i=1, typecount
            do j=1, typecount
                do k=1, typecount
                    do l=k, typecount
                        !Write partial bond torsion labels
                        write(54,35) frame,i,j,k,l,count2neg(i,j,k,l),btors_neg_part_mean(i,j,k,l), btors_neg_part_stdev(i,j,k,l),&
                                count2pos(i,j,k,l),btors_pos_part_mean(i,j,k,l), btors_pos_part_stdev(i,j,k,l)
                        35  format(i11,4(',',9x,i2),2(',',i11,',',f11.6,',',f11.6),$)
                    
                        do m=1, 180
                            write(54,40) btors_neg_part(i,j,k,l,m),btors_pos_part(i,j,k,l,m)
                            40  format(',',f11.6,',',f11.6,$)
                        end do
                        write(54,*)  !New line
                    end do
                end do
            end do
        end do
    end if


    DEALLOCATE(btors_pos_part)
    DEALLOCATE(btors_pos_part_mean)
    DEALLOCATE(btors_pos_part_stdev)
    DEALLOCATE(btors_neg_part)
    DEALLOCATE(btors_neg_part_mean)
    DEALLOCATE(btors_neg_part_stdev)


    return
end 


!END OF CAL_BTORSION ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_SC ROUTINE
!Self-similarity search
!Define SC based on mode
!in_sc_cells > 0 - Use internally defined SC (no. of SC = in_sc_cells)
!in_sc_cells = 0 - Read in SC from file (no. of SC define in file)
!in_sc_cells =-1 - Use self-similiarty search to find SC in xyz file
!                  For multi-frame xyz, SC found in previous frame are
!                  added in the search in the next frame.  This ensures
!                  unique SC classiication amongst all frames.

subroutine CAL_SC
    
    use KINDS, only: dp  
    use VARIABLES, only: flag_surf, frame,                             &
    in_sc_cells, in_sc_cutoff, in_sc_res, in_sc_labs, natoms 
    IMPLICIT NONE

        
    !Local     
    integer, parameter :: con_scmax = 2000                              !maximum number of SC
    integer, parameter :: con_pamax = 100                               !maximum number of particle in a signature cell
    integer, parameter :: con_nnmax = 1000                              !maximum number of particles in nn list used for fitting

    integer :: particle                                                 !particle label
    integer :: cell                                                     !SC label
    integer :: cell_total                                               !total SCs 
    integer :: sc_res                                                   !angular resolution for x,y,z axial rotations
    real(dp) :: sc_cutoff                                               !minimal separations squared per nn (SS/nn) cutoff value below which search stops
    integer :: sc_depth                                                 !nn depth of SC within loop (1st nn or 1st+2nd nn or more etc)
    integer :: sc_type                                                  !surf, bulk or all classification of SC within loop
    
    
    integer :: i, j, it1, it2
    real(dp) :: tsin(360), tcos(360)                                    !cosine and sine look up tables 
    
    integer :: sc_pt                                                    !number of nn used in fit for single particle 
    real(dp) :: sc_p(con_nnmax,3)                                       !x,y,z positions of nn used for single particle
    integer :: sc_ct(con_scmax)                                         !total particles SC of type i      
    real(dp) :: sc_c(con_scmax,con_pamax,3)                             !SC coordinate of type i, particle j, coordinate k
    
    integer :: sc_dep(con_scmax)                                        !nearest neighbor depth (SC uses 1st, 2nd or high nn)
    integer :: sc_typ(con_scmax)                                        !SC type (1=all, 2=bulk only, 3=surface only) 
    
    real(dp) :: angle(3)                                                !x,y,z axial rotations ouput from subroutine
    real(dp) :: score                                                   !fitness score from subroutine (average per particle separation)
    
    real(dp) :: sc_orient(natoms(frame),con_scmax,3)                    !x,y,z axial rotation angles of SC for atom i, SC type j, axial direction k 
    real(dp) :: sc_score(natoms(frame),con_scmax)                       !Fitness value for particle i fitting SC type j
    real(dp) :: temp_minimum
    integer :: sc_similar(natoms(frame))
    integer :: sc_classify(natoms(frame))                               !Classified SC type of particle i
    
    integer :: track, track_count
    integer :: track_conv(1000)   !fix
    
    
    
    !Angular resolution for axial rotations (integer)
    sc_res = in_sc_res
    
    !Minimal separations squared (SS) per nn cutoff value below which search stops  
    sc_cutoff = in_sc_cutoff   
    
    !Initialize arrays
    sc_orient = 0.0_dp
    sc_score  = 2.0_dp                                                  !single SC particle can be maximum of 2A away from system particle
    sc_similar = 0
    
    !Calculate sin and cos tables for speed
    call CAL_SC_TABLE(tsin,tcos)
    
    
    
    
    
    
    !Create SC lists for fitting
    
    !Gather SC from internal list
    if(in_sc_cells>0) then     
        call CAL_SC_CELLS_INTERNAL(con_scmax,con_pamax,sc_ct,sc_c,sc_dep,sc_typ)
        cell_total = in_sc_cells
    end if
    
    !Gather SC from read in of input file
    if(in_sc_cells==0) then
        !    call CAL_SC_CELLS_READIN
    end if
    
    !Gather SC from self-similarity search from xyz file       
    if(in_sc_cells==-1) then
        
        sc_depth = 1
        cell_total = natoms(frame)
        
        do particle=1, natoms(frame)
            call CAL_SC_GETNN_ALL(con_nnmax,sc_depth,particle,sc_pt,sc_p)
            
            sc_dep(particle) = sc_depth
            sc_typ(particle) = 0
            sc_ct(particle) = sc_pt
            do i=1, sc_pt
                sc_c(particle,i,1) = sc_p(i,1)
                sc_c(particle,i,2) = sc_p(i,2)
                sc_c(particle,i,3) = sc_p(i,3)
            end do
            
        end do
    end if
    
    
    
    
    
    
    
    !Output of used SC for visualization 
    open(10,status="unknown",file='ov_SC_cells.xyz')
    
    do i=1, cell_total
        write(10,'(i10,/)') sc_ct(i)+1    
        do j=1, sc_ct(i)
            write(10,'(a,2x,3(F10.5,2X),i3)') 'AA',sc_c(i,j,1),sc_c(i,j,2),sc_c(i,j,3)
        end do
        write(10,20)
        20  format('AA  0.000 0.000 0.000')
        
    end do  
    
    close(10)      
    
    
    
    !Main search loop  
    do particle=1, natoms(frame)
    
        print*,'Particle:',particle
        
        if(in_sc_cells==-1) then
            if(sc_similar(particle)>0) then
                CYCLE
            end if
        end if
        
        
        do i=1, cell_total
            
            !Use SC list from input file        
            if(in_sc_cells>0) then
                cell=in_sc_labs(i)
            end if
            
            !Use SC list from all atoms
            if(in_sc_cells==-1) then
                cell = i
                
                if(sc_similar(i)>0) then
                    CYCLE
                end if
                
                
            end if
            
            
            
            !MAKE NN LIST FOR FITTING                    
            !Build list of nn based on SC nn depth and if SC is surface, bulk etc    
            sc_depth = sc_dep(cell)                                     !sc nn depths 
            sc_type  = sc_typ(cell)                                     !sc type class (0=all,1=bulk,2=surface)
            
            
            !--for type all SC, on any particles, use all n list         
            if(sc_type==0) then 
                call CAL_SC_GETNN_ALL(con_nnmax,sc_depth,particle,sc_pt,sc_p)           
            end if
            
            !--for bulk SC, on surface particle, skip fitting          
            if(sc_type==1.and.flag_surf(particle)==1) then
                CYCLE                                           
            end if
            
            !--for bulk SC, on non-surface particle (bulk), use all n list           
            if(sc_type==1.and.flag_surf(particle)==0) then          
                call CAL_SC_GETNN_ALL(con_nnmax,sc_depth,particle,sc_pt,sc_p)           
            end if
            
            !--for surf SC, on surface particle, use surf n list
            if(sc_type==2.and.flag_surf(particle)==1) then          
                call CAL_SC_GETNN_SURF(con_nnmax,sc_depth,particle,sc_pt,sc_p)           
            end if       
            
            !--for surf SC, on non-surface particle, skip fitting
            if(sc_type==2.and.flag_surf(particle)==0) then          
                CYCLE          
            end if                
            
            
            !FIT SC TO LOCAL NN ENVIRONMENT
            !--particle in SC must be equal to nn being fitted      
            score = 2.0_dp
            if(sc_pt==sc_ct(cell)) then                               !replaced eq with ge to allow for SC plus additions
                
                call CAL_SC_FIT(con_scmax,con_pamax,con_nnmax,cell,sc_res, &
                sc_ct,sc_c,sc_pt,sc_p,angle,tsin,tcos,score)
                
                !--record axial orientation and SS value of fit for atom i and SC type j            
                sc_orient(particle,cell,:) = angle(:)
                sc_score(particle,cell) = score
                
            end if 
            
            
            !Search mode
            if(in_sc_cells==-1) then
                if(score<in_sc_cutoff) then
                    sc_similar(particle) = i
                    sc_similar(i) = particle
                end if
            end if
            
            !Exit search loop if fit found (speed up technique)            
            if(in_sc_cells>0) then
                if(score<in_sc_cutoff) then
                    sc_similar(particle) = i
                    EXIT
                end if
            end if
            
            
            
        end do
        
        
!       -Find minimum value from all the SCs
!        temp_minimum = 2.0
!        do i=1, in_sc_cells
!            cell=in_sc_labs(i)
!            if(sc_score(particle,cell)<in_sc_cutoff) then
!            
!                if(sc_score(particle,cell)<temp_minimum) then
!                    temp_minimum = sc_score(particle,cell)
!                    sc_similar(particle) = cell
!                end if
!                
!            end if
!        end do
        
        
!        print*,'SC',frame,particle, sc_score(particle,1),sc_score(particle,2),sc_score(particle,3),sc_score(particle,4)   
        
    end do
    
    
    if(in_sc_cells==-1) then 
        
        !Remove redundant labels in similar array      
        do i=1, natoms(frame)
            if(sc_similar(i)>i) sc_similar(i) = i
        end do
        
        !Find unique SC       
        track = 1
        track_count = 1
        track_conv(1) = 1
        do i=1, natoms(frame)
            if(sc_similar(i)>track) then
                
                track = sc_similar(i)
                track_count = track_count + 1
                
                track_conv(track_count) = track
                
                
      !          print*,'i,track:',i,track,track_count
            end if
            
        end do
        
        !Create sc_classify array
        do i=1, natoms(frame)
            
            it1 = sc_similar(i)
            
            do j=1, track_count
                it2 = track_conv(j)
                
                if(it1==it2) then
                    sc_classify(i) = j
                end if
                
            end do
            
            print*,'i,C:',i,sc_similar(i),sc_classify(i)
            
        end do
        
        
    end if
    
    !CHECK - quick add
    if(in_sc_cells>0) then 
        do i=1, natoms(frame)
            it1 = sc_similar(i)
            
            sc_classify(i) = sc_similar(i)
        
        end do
    end if
    
    
    
    !!Check for degenerate orientations (dif. orientation with same cost) for same types and set to one orientation
    !      do particle=1, natoms(frame)
    !        if(particle/=0) then       !if particle is classified
    !          
    !        
    !        
    !
    !        
    !        
    !        end if
    !      end do
    
    
    
    !Write classified config to xyz file      
    call CAL_SC_XYZ(con_scmax,sc_classify,sc_score,sc_orient,tsin,tcos)
    
    
    
    close(10)         
    
    
    
    return
end 


!END OF CAL_SC ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_SC_TABLE ROUTINE
!Produces an integer resolution list of values for the sin 
!and cos functions

subroutine CAL_SC_TABLE(tsin,tcos)
    
    use KINDS
    IMPLICIT NONE
    
    !Passed 
    real(dp), intent(out) :: tsin(360), tcos(360)            
    
    !Local     
    integer :: i
    real(dp) angle
    
    angle = 1.0/57.295779513
    
    do i=1, 360
        tsin(i) = sin(angle*i)
        tcos(i) = cos(angle*i)
    end do
    
    
    return
end 


!END OF CAL_SC_TABLE ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_SC_CELLS_INTERNAL ROUTINE
!-Signature Cells (SC) use nearest neighbour (nn) cages around a
! particle.  Can use 1st nn or 1st and 2nd nn together for fitting
!-SC have their origin at x=y=z=0 and thus can have positive and 
! negative coordinates.  Centre particle never included.  
!-Coordinates are fractional normalized to 1 for the largest nn
! distance.  In the case of using 1st and 2nd nn, this would be a 
! 2nd nn distance.

! sc_c(i,j,k) - bulk SC coordinate k of type i, particle j where
!               for k, 1=x,2=y,3=z.   


!BULK CELLS
!1 = FCC
!2 = HCP
!3 = ICOS
!4 = TWISTED ICOS 

subroutine CAL_SC_CELLS_INTERNAL(con_scmax,con_pamax,sc_ct,sc_c,sc_dep,sc_typ)
       
    use KINDS   
    IMPLICIT NONE
    
    !Passed      
    integer, intent(in) :: con_scmax
    integer, intent(in) :: con_pamax
    integer, intent(out) :: sc_ct(con_scmax)  
    real(dp), intent(out) :: sc_c(con_scmax,con_pamax,3) 
    integer, intent(out) :: sc_dep(con_scmax)
    integer, intent(out) :: sc_typ(con_scmax)                           !type classification of SC (0=all,1=bulk,2=surface)                                         
    
    !Local     
    integer :: c                                                        !variable for SC type
    
    
    
    !List of bulk signature cells
    
    !Type 1 - FCC
    c = 1
    sc_dep(c)    = 1
    sc_typ(c)    = 1
    sc_ct(c)     = 12
    sc_c(c,1,1)  = -0.514258006; sc_c(c,1,2)  = -0.807588065; sc_c(c,1,3)  = -0.025036245
    sc_c(c,2,1)  = -0.822136154; sc_c(c,2,2)  = -0.019284675; sc_c(c,2,3)  =  0.489560088
    sc_c(c,3,1)  = -0.808264721; sc_c(c,3,2)  = -0.021652969; sc_c(c,3,3)  = -0.502078211
    sc_c(c,4,1)  =  0.282841903; sc_c(c,4,2)  = -0.807926393; sc_c(c,4,3)  =  0.465538826
    sc_c(c,5,1)  =  0.279796954; sc_c(c,5,2)  = -0.804543117; sc_c(c,5,3)  = -0.510536402
    sc_c(c,6,1)  = -0.029096177; sc_c(c,6,2)  = -0.012518123; sc_c(c,6,3)  =  0.976075228
    sc_c(c,7,1)  = -0.329869444; sc_c(c,7,2)  =  0.777815233; sc_c(c,7,3)  =  0.489560088
    sc_c(c,8,1)  = -0.020299658; sc_c(c,8,2)  = -0.008458191; sc_c(c,8,3)  = -0.999758162
    sc_c(c,9,1)  = -0.314306373; sc_c(c,9,2)  =  0.771048681; sc_c(c,9,3)  = -0.509859746
    sc_c(c,10,1) =  0.810971342; sc_c(c,10,2) =  0.000676655; sc_c(c,10,3) =  0.479071932
    sc_c(c,11,1) =  0.806573082; sc_c(c,11,2) =  0.010149829; sc_c(c,11,3) = -0.501739883
    sc_c(c,12,1) =  0.511213057; sc_c(c,12,2) =  0.798791547; sc_c(c,12,3) = -0.018946348
    
    !Type 2 - HCP
    c = 2  
    sc_dep(c)    = 1    
    sc_typ(c)    = 1      
    sc_ct(c)     = 12
    sc_c(c,1,1)  = -0.811860597; sc_c(c,1,2)  = -0.001366194; sc_c(c,1,3)  =  0.487731145
    sc_c(c,2,1)  = -0.809469758; sc_c(c,2,2)  = -0.007172517; sc_c(c,2,3)  = -0.489780436
    sc_c(c,3,1)  = -0.018443615; sc_c(c,3,2)  = -0.006147872; sc_c(c,3,3)  =  0.978536226
    sc_c(c,4,1)  = -0.017077421; sc_c(c,4,2)  =  0.001366194; sc_c(c,4,3)  = -0.979219323
    sc_c(c,5,1)  = -0.311150612; sc_c(c,5,2)  =  0.795807821; sc_c(c,5,3)  =  0.519495148
    sc_c(c,6,1)  = -0.303294998; sc_c(c,6,2)  = -0.800931047; sc_c(c,6,3)  =  0.501393082
    sc_c(c,7,1)  = -0.297147126; sc_c(c,7,2)  =  0.793416982; sc_c(c,7,3)  = -0.481583274
    sc_c(c,8,1)  = -0.300904159; sc_c(c,8,2)  = -0.800589499; sc_c(c,8,3)  = -0.488072694
    sc_c(c,9,1)  =  0.804346531; sc_c(c,9,2)  = -0.000341548; sc_c(c,9,3)  =  0.49524521
    sc_c(c,10,1) =  0.801272596; sc_c(c,10,2) =  0.003073936; sc_c(c,10,3) = -0.490121984
    sc_c(c,11,1) =  0.518812051; sc_c(c,11,2) =  0.815276081; sc_c(c,11,3) =  0.025274583
    sc_c(c,12,1) =  0.526326117; sc_c(c,12,2) = -0.811519048; sc_c(c,12,3) = -0.002049291      
    
    !Type 3 - Icosahedral
    c = 3
    sc_dep(c)    = 1 
    sc_typ(c)    = 1
    sc_ct(c)     = 12
    sc_c(c,1,1)  =  1.000000000; sc_c(c,1,2)  =  0.000000000; sc_c(c,1,3)  =  0.000000000
    sc_c(c,2,1)  =  0.447213595; sc_c(c,2,2)  =  0.894427191; sc_c(c,2,3)  =  0.000000000
    sc_c(c,3,1)  =  0.447213595; sc_c(c,3,2)  =  0.276393202; sc_c(c,3,3)  =  0.850650808
    sc_c(c,4,1)  =  0.447213595; sc_c(c,4,2)  = -0.723606798; sc_c(c,4,3)  =  0.525731112
    sc_c(c,5,1)  =  0.447213595; sc_c(c,5,2)  = -0.723606798; sc_c(c,5,3)  = -0.525731112
    sc_c(c,6,1)  =  0.447213595; sc_c(c,6,2)  =  0.276393202; sc_c(c,6,3)  = -0.850650808
    sc_c(c,7,1)  = -1.000000000; sc_c(c,7,2)  =  0.000000000; sc_c(c,7,3)  =  0.000000000
    sc_c(c,8,1)  = -0.447213595; sc_c(c,8,2)  = -0.894427191; sc_c(c,8,3)  =  0.000000000
    sc_c(c,9,1)  = -0.447213595; sc_c(c,9,2)  = -0.276393202; sc_c(c,9,3)  = -0.850650808
    sc_c(c,10,1) = -0.447213595; sc_c(c,10,2) =  0.723606798; sc_c(c,10,3) = -0.525731112
    sc_c(c,11,1) = -0.447213595; sc_c(c,11,2) =  0.723606798; sc_c(c,11,3) =  0.525731112
    sc_c(c,12,1) = -0.447213595; sc_c(c,12,2) = -0.276393202; sc_c(c,12,3) =  0.850650808
    
    !Type 4 - Twisted Icosahedral
    c = 4
    sc_dep(c)    = 1 
    sc_typ(c)    = 1
    sc_ct(c)     = 12
    sc_c(c,1,1)  = -0.799773855; sc_c(c,1,2)  = -0.015373547; sc_c(c,1,3)  = -0.506977660
    sc_c(c,2,1)  =  0.002445792; sc_c(c,2,2)  = -0.005939780; sc_c(c,2,3)  = -0.999979368
    sc_c(c,3,1)  = -0.304326356; sc_c(c,3,2)  =  0.790689487; sc_c(c,3,3)  = -0.496845095
    sc_c(c,4,1)  = -0.808858224; sc_c(c,4,2)  =  0.480423351; sc_c(c,4,3)  =  0.296988981
    sc_c(c,5,1)  = -0.800472653; sc_c(c,5,2)  = -0.504531869; sc_c(c,5,3)  =  0.299085374
    sc_c(c,6,1)  = -0.300832368; sc_c(c,6,2)  = -0.808159427; sc_c(c,6,3)  = -0.494399303
    sc_c(c,7,1)  =  0.811653415; sc_c(c,7,2)  = -0.004891583; sc_c(c,7,3)  =  0.502086077
    sc_c(c,8,1)  =  0.830520950; sc_c(c,8,2)  =  0.001397595; sc_c(c,8,3)  = -0.498941488
    sc_c(c,9,1)  =  0.512218642; sc_c(c,9,2)  =  0.815496801; sc_c(c,9,3)  =  0.032494088
    sc_c(c,10,1) = -0.000349399; sc_c(c,10,2) =  0.486712530; sc_c(c,10,3) =  0.838906521
    sc_c(c,11,1) =  0.018867535; sc_c(c,11,2) = -0.510821047; sc_c(c,11,3) =  0.826328164
    sc_c(c,12,1) =  0.520254815; sc_c(c,12,2) = -0.817942593; sc_c(c,12,3) =  0.013975952
    
    !Type 5 - Outer atom icosahedral cage to 2nd nn (boron)
    c = 5
    sc_dep(c)    = 2  
    sc_typ(c)    = 0
    sc_ct(c)     = 10
    sc_c(c,1,1)  = -0.32492; sc_c(c,1,2)  =  0.52573; sc_c(c,1,3)  =  0.000000000
    sc_c(c,2,1)  = -0.32492; sc_c(c,2,2)  =  0.16246; sc_c(c,2,3)  =  0.500000000
    sc_c(c,3,1)  = -0.32492; sc_c(c,3,2)  = -0.42533; sc_c(c,3,3)  =  0.30902
    sc_c(c,4,1)  = -0.32492; sc_c(c,4,2)  = -0.42533; sc_c(c,4,3)  = -0.30902
    sc_c(c,5,1)  = -0.32492; sc_c(c,5,2)  =  0.16246; sc_c(c,5,3)  = -0.50000
    sc_c(c,6,1)  = -0.85065; sc_c(c,6,2)  = -0.52573; sc_c(c,6,3)  =  0.00000
    sc_c(c,7,1)  = -0.85065; sc_c(c,7,2)  = -0.16246; sc_c(c,7,3)  = -0.50000
    sc_c(c,8,1)  = -0.85065; sc_c(c,8,2)  =  0.42533; sc_c(c,8,3)  = -0.30902
    sc_c(c,9,1)  = -0.85065; sc_c(c,9,2)  =  0.42533; sc_c(c,9,3)  =  0.30902
    sc_c(c,10,1) = -0.85065; sc_c(c,10,2) = -0.16246; sc_c(c,10,3) =  0.50000  
    
    !Type 6 - Outer atom icosahedral cage to 1st nn (boron)
    c = 6
    sc_dep(c)    = 1    
    sc_typ(c)    = 0
    sc_ct(c)     = 5
    sc_c(c,1,1)  = -0.52573; sc_c(c,1,2)  =  0.85064; sc_c(c,1,3)  =  0.000000000
    sc_c(c,2,1)  = -0.52573; sc_c(c,2,2)  =  0.26286; sc_c(c,2,3)  =  0.80901
    sc_c(c,3,1)  = -0.52573; sc_c(c,3,2)  = -0.68819; sc_c(c,3,3)  =  0.50000
    sc_c(c,4,1)  = -0.52573; sc_c(c,4,2)  = -0.68819; sc_c(c,4,3)  = -0.50000
    sc_c(c,5,1)  = -0.52573; sc_c(c,5,2)  =  0.26289; sc_c(c,5,3)  = -0.80901    
    
    
    
    !Type 7 - Diamond FCC (1st+2nd nn)
    c = 7
    sc_dep(c)    = 2    
    sc_typ(c)    = 1
    sc_ct(c)     = 16
    sc_c(c,1,1)  =  0.57854; sc_c(c,1,2)  = -0.19021; sc_c(c,1,3)  =  0.02536
    sc_c(c,2,1)  = -0.09419; sc_c(c,2,2)  =  0.29156; sc_c(c,2,3)  = -0.52366
    sc_c(c,3,1)  = -0.11023; sc_c(c,3,2)  =  0.37710; sc_c(c,3,3)  =  0.46379 
    sc_c(c,4,1)  = -0.37652; sc_c(c,4,2)  = -0.47655; sc_c(c,4,3)  =  0.03530
    sc_c(c,5,1)  =  0.95883; sc_c(c,5,2)  =  0.28196; sc_c(c,5,3)  = -0.01022 
    sc_c(c,6,1)  =  0.68951; sc_c(c,6,2)  = -0.56677; sc_c(c,6,3)  = -0.43729
    sc_c(c,7,1)  =  0.67358; sc_c(c,7,2)  = -0.48167; sc_c(c,7,3)  =  0.54908
    sc_c(c,8,1)  =  0.01608; sc_c(c,8,2)  = -0.08482; sc_c(c,8,3)  = -0.98673 
    sc_c(c,9,1)  =  0.28471; sc_c(c,9,2)  =  0.76765; sc_c(c,9,3)  = -0.55888
    sc_c(c,10,1) = -0.67502; sc_c(c,10,2) =  0.47791; sc_c(c,10,3) = -0.54934
    sc_c(c,11,1) =  0.27067; sc_c(c,11,2) =  0.84862; sc_c(c,11,3) =  0.42359
    sc_c(c,12,1) = -0.01460; sc_c(c,12,2) =  0.08257; sc_c(c,12,3) =  0.98289    
    sc_c(c,13,1) = -0.69012; sc_c(c,13,2) =  0.55908; sc_c(c,13,3) =  0.43377
    sc_c(c,14,1) = -0.26604; sc_c(c,14,2) = -0.85191; sc_c(c,14,3) = -0.42802
    sc_c(c,15,1) = -0.28239; sc_c(c,15,2) = -0.76649; sc_c(c,15,3) =  0.55933
    sc_c(c,16,1) = -0.95602; sc_c(c,16,2) = -0.29316; sc_c(c,16,3) =  0.00953   
    
    !Type 8 - Diamond HCP (1st+2nd nn)  
    c = 8
    sc_dep(c)    = 2
    sc_typ(c)    = 1
    sc_ct(c)     = 16
    sc_c(c,1,1)  = -0.10582; sc_c(c,1,2)  = -0.37305; sc_c(c,1,3)  = -0.46669
    sc_c(c,2,1)  = -0.35294; sc_c(c,2,2)  =  0.50119; sc_c(c,2,3)  = -0.04878
    sc_c(c,3,1)  = -0.12203; sc_c(c,3,2)  = -0.28859; sc_c(c,3,3)  =  0.51943 
    sc_c(c,4,1)  =  0.58404; sc_c(c,4,2)  =  0.16062; sc_c(c,4,3)  = -0.00386
    sc_c(c,5,1)  = -0.69170; sc_c(c,5,2)  = -0.53317; sc_c(c,5,3)  = -0.46253
    sc_c(c,6,1)  =  0.01613; sc_c(c,6,2)  = -0.08464; sc_c(c,6,3)  = -0.98641
    sc_c(c,7,1)  =  0.24954; sc_c(c,7,2)  = -0.86589; sc_c(c,7,3)  = -0.41858
    sc_c(c,8,1)  = -0.69556; sc_c(c,8,2)  =  0.46442; sc_c(c,8,3)  = -0.54819 
    sc_c(c,9,1)  = -0.00306; sc_c(c,9,2)  =  0.99433; sc_c(c,9,3)  = -0.08500 
    sc_c(c,10,1) = -0.71187; sc_c(c,10,2) =  0.54901; sc_c(c,10,3) =  0.43791 
    sc_c(c,11,1) = -0.70779; sc_c(c,11,2) = -0.44841; sc_c(c,11,3) =  0.52229
    sc_c(c,12,1) = -0.01656; sc_c(c,12,2) =  0.08437; sc_c(c,12,3) =  0.98620 
    sc_c(c,13,1) =  0.23328; sc_c(c,13,2) = -0.78132; sc_c(c,13,3) =  0.56795 
    sc_c(c,14,1) =  0.70508; sc_c(c,14,2) =  0.44744; sc_c(c,14,3) = -0.52383
    sc_c(c,15,1) =  0.68879; sc_c(c,15,2) =  0.53195; sc_c(c,15,3) =  0.46364 
    sc_c(c,16,1) =  0.94027; sc_c(c,16,2) = -0.32862; sc_c(c,16,3) =  0.04399    
    
    !Type 9 - Diamond Backbone (1st+2nd nn)  
    c = 9
    sc_dep(c)    = 2  
    sc_typ(c)    = 1
    sc_ct(c)     = 16
    sc_c(c,1,1)  = -0.09440; sc_c(c,1,2)  =  0.29067; sc_c(c,1,3)  = -0.52832
    sc_c(c,2,1)  = -0.36747; sc_c(c,2,2)  = -0.48700; sc_c(c,2,3)  =  0.03578 
    sc_c(c,3,1)  = -0.11073; sc_c(c,3,2)  =  0.37628; sc_c(c,3,3)  =  0.46761 
    sc_c(c,4,1)  =  0.57787; sc_c(c,4,2)  = -0.19712; sc_c(c,4,3)  =  0.02636 
    sc_c(c,5,1)  =  0.28391; sc_c(c,5,2)  =  0.76602; sc_c(c,5,3)  = -0.56293
    sc_c(c,6,1)  =  0.01629; sc_c(c,6,2)  = -0.08547; sc_c(c,6,3)  = -0.99608
    sc_c(c,7,1)  = -0.67468; sc_c(c,7,2)  =  0.47207; sc_c(c,7,3)  = -0.55338
    sc_c(c,8,1)  = -0.70936; sc_c(c,8,2)  = -0.52433; sc_c(c,8,3)  = -0.46844 
    sc_c(c,9,1)  = -0.72580; sc_c(c,9,2)  = -0.43886; sc_c(c,9,3)  =  0.52751
    sc_c(c,10,1) = -0.01671; sc_c(c,10,2) = -0.98510; sc_c(c,10,3) =  0.08426
    sc_c(c,11,1) =  0.26762; sc_c(c,11,2) =  0.85172; sc_c(c,11,3) =  0.43275
    sc_c(c,12,1) = -0.01637; sc_c(c,12,2) =  0.08574; sc_c(c,12,3) =  0.99618
    sc_c(c,13,1) = -0.69110; sc_c(c,13,2) =  0.55773; sc_c(c,13,3) =  0.44230
    sc_c(c,14,1) =  0.87267; sc_c(c,14,2) = -0.03921; sc_c(c,14,3) = -0.48420
    sc_c(c,15,1) =  0.85652; sc_c(c,15,2) =  0.04635; sc_c(c,15,3) =  0.51174
    sc_c(c,16,1) =  0.56754; sc_c(c,16,2) = -0.80594; sc_c(c,16,3) =  0.07844       
    
    
    !Type 41 - 100 Surface   
    c = 41 
    sc_dep(c)   = 1     
    sc_typ(c)   = 2
    sc_ct(c)    = 4
    sc_c(c,1,1) =  1.000000; sc_c(c,1,2) =  0.000000; sc_c(c,1,3) =  0.000000
    sc_c(c,2,1) = -1.000000; sc_c(c,2,2) =  0.000000; sc_c(c,2,3) =  0.000000 
    sc_c(c,3,1) =  0.000000; sc_c(c,3,2) =  1.000000; sc_c(c,3,3) =  0.000000 
    sc_c(c,4,1) =  0.000000; sc_c(c,4,2) = -1.000000; sc_c(c,4,3) =  0.000000
    
    
    !Type 42 - 111 Surface 
    c = 42    
    sc_dep(c)   = 1
    sc_typ(c)   = 2
    sc_ct(c)    = 6
    sc_c(c,1,1) =  1.000000; sc_c(c,1,2) =  0.000000; sc_c(c,1,3) =  0.000000
    sc_c(c,2,1) = -1.000000; sc_c(c,2,2) =  0.000000; sc_c(c,2,3) =  0.000000 
    sc_c(c,3,1) =  0.500000; sc_c(c,3,2) =  0.866025; sc_c(c,3,3) =  0.000000
    sc_c(c,4,1) =  0.500000; sc_c(c,4,2) = -0.866025; sc_c(c,4,3) =  0.000000
    sc_c(c,5,1) = -0.500000; sc_c(c,5,2) =  0.866025; sc_c(c,5,3) =  0.000000 
    sc_c(c,6,1) = -0.500000; sc_c(c,6,2) = -0.866025; sc_c(c,6,3) =  0.000000
    
    !Type 43 - 110 Surface 
    c = 43   
    sc_dep(c)  = 1  
    sc_typ(c)  = 2
    sc_ct(c)   = 6
    sc_c(c,1,1) = -0.514258006; sc_c(c,1,2) = -0.807588065; sc_c(c,1,3) = -0.025036245
    sc_c(c,2,1) = -0.822136154; sc_c(c,2,2) = -0.019284675; sc_c(c,2,3) =  0.489560088
    sc_c(c,3,1) =  0.279796954; sc_c(c,3,2) = -0.804543117; sc_c(c,3,3) = -0.510536402
    sc_c(c,4,1) = -0.029096177; sc_c(c,4,2) = -0.012518123; sc_c(c,4,3) =  0.976075228
    sc_c(c,5,1) =  0.810971342; sc_c(c,5,2) =  0.000676655; sc_c(c,5,3) =  0.479071932
    sc_c(c,6,1) =  0.806573082; sc_c(c,6,2) =  0.010149829; sc_c(c,6,3) = -0.501739883
    
    

    return
end 


!END OF CAL_SC_CELLS_INTERNAL ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_SC_GETNN_ALL ROUTINE
!Build list of nn for fitting of particle and record the nn 
!positions relative to centre particle (particle) in sc_p. Used
!entire neighbour list  

SUBROUTINE CAL_SC_GETNN_ALL(con_nnmax,sc_depth,particle,sc_pt,sc_p) 
 
    use KINDS, only: dp
    use VARIABLES, only: ncord, nnlist, xc, yc, zc,                    &
    xl, yl, zl, xl2inv, yl2inv, zl2inv
    IMPLICIT NONE
    
    !Passed
    integer :: con_nnmax
    integer :: sc_depth 
    integer :: particle
    integer :: sc_pt
    real(dp) sc_p(con_nnmax,3)      
    
    !Local       
    integer :: i, j, k, l
    integer :: it, it2
    integer :: duplicate_flag
    integer :: temp_sum, temp_list(con_nnmax)
    real(dp) dx, dy, dz, dr, dr_max
    
    
    
    !1st nn added to list (either surface or bulk nn) 
    temp_sum = 0
    do i=1, ncord(particle)
        it = nnlist(particle,i)
        temp_sum = temp_sum+1
        temp_list(temp_sum) = it
    end do
    
    !2nd nn and more counted below
    do i=2, sc_depth 
        
        do j=1, temp_sum
            it = temp_list(j)
            do k=1, ncord(it)
                it2 = nnlist(it,k) 
                !Check so that it2 is not in existing list (no duplicates)            
                duplicate_flag = 0
                do l=1, temp_sum
                    if(it2==temp_list(l)) duplicate_flag = 1
                end do
                
                if(duplicate_flag==0.and.it2/=particle) then
                    temp_sum= temp_sum+1
                    temp_list(temp_sum) = it2
                end if
                
            end do        
        end do
    end do        
    
    !Find largest relative positions of nn to centre atom
    dr_max = 0.0
    do i=1, temp_sum
        it = temp_list(i)  
        dx = xc(it) - xc(particle)
        dy = yc(it) - yc(particle)
        dz = zc(it) - zc(particle)   
        
        dx = dx-(int(dx*xl2inv)*xl)
        dy = dy-(int(dy*yl2inv)*yl)
        dz = dz-(int(dz*zl2inv)*zl)
        
        dr = sqrt(dx*dx+dy*dy+dz*dz)
        
        if(dr>dr_max) dr_max = dr
    end do
    
    !Normalize relative positions 
    do i=1, temp_sum
        it = temp_list(i)
        
        dx = xc(it) - xc(particle)
        dy = yc(it) - yc(particle)
        dz = zc(it) - zc(particle)   
        
        dx = dx-(int(dx*xl2inv)*xl)
        dy = dy-(int(dy*yl2inv)*yl)
        dz = dz-(int(dz*zl2inv)*zl)        
        
        dx = dx/dr_max
        dy = dy/dr_max
        dz = dz/dr_max
        
        dr = sqrt(dx*dx+dy*dy+dz*dz)
        
        sc_p(i,1) = dx
        sc_p(i,2) = dy
        sc_p(i,3) = dz     
        
    end do
    
    sc_pt = temp_sum
    
    return
end 


!END OF CAL_SC_GETNN_ALL ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_SC_GETNN_SURF ROUTINE
!Build list of nn for fitting of particle and record the nn 
!positions relative to centre particle (particle) in sc_p. Uses 
!the surface layer neighbour list 

SUBROUTINE CAL_SC_GETNN_SURF(con_nnmax,sc_depth,particle,sc_pt,sc_p) 
    
    use KINDS, only: dp
    use VARIABLES, only: ncordS, nnlistS, xc, yc, zc,                  &
    xl, yl, zl, xl2inv, yl2inv, zl2inv    
    IMPLICIT NONE
    
    !Passed 
    integer :: con_nnmax
    integer :: sc_depth
    integer :: particle
    integer :: sc_pt
    real(dp) sc_p(con_nnmax,3)      
    
    !Local      
    integer :: i, j, k, l
    integer :: it, it2
    integer :: duplicate_flag
    integer :: temp_sum, temp_list(con_nnmax)
    real(dp) dx, dy, dz, dr, dr_max
    
    
    !Make list of 1st nn
    temp_sum = 0
    do i=1, ncordS(particle)
        it = nnlistS(particle,i)
        temp_sum = temp_sum+1
        temp_list(temp_sum) = it
    end do            
    
    !2nd nn and more counted below
    do i=2, sc_depth 
        
        do j=1, temp_sum
            it = temp_list(j)
            do k=1, ncordS(it)
                it2 = nnlistS(it,k) 
                !Check so that it2 is not in existing list (no duplicates)            
                duplicate_flag = 0
                do l=1, temp_sum
                    if(it2==temp_list(l)) duplicate_flag = 1
                end do
                
                if(duplicate_flag==0.and.it2/=particle) then
                    temp_sum= temp_sum+1
                    temp_list(temp_sum) = it2
                end if
                
            end do        
        end do
    end do
    
    !Find largest relative positions of nn to centre atom
    dr_max = 0.0
    do i=1, temp_sum
        it = temp_list(i)  
        dx = xc(it) - xc(particle)
        dy = yc(it) - yc(particle)
        dz = zc(it) - zc(particle)   
        
        dx = dx-(int(dx*xl2inv)*xl)
        dy = dy-(int(dy*yl2inv)*yl)
        dz = dz-(int(dz*zl2inv)*zl)
        
        dr = sqrt(dx*dx+dy*dy+dz*dz)
        
        if(dr>dr_max) dr_max = dr
    end do
    
    !Normalize relative positions 
    do i=1, temp_sum
        it = temp_list(i)
        
        dx = xc(it) - xc(particle)
        dy = yc(it) - yc(particle)
        dz = zc(it) - zc(particle)   
        
        dx = dx-(int(dx*xl2inv)*xl)
        dy = dy-(int(dy*yl2inv)*yl)
        dz = dz-(int(dz*zl2inv)*zl)        
        
        dx = dx/dr_max
        dy = dy/dr_max
        dz = dz/dr_max
        
        dr = sqrt(dx*dx+dy*dy+dz*dz)
        
        sc_p(i,1) = dx
        sc_p(i,2) = dy
        sc_p(i,3) = dz     
        
    end do
    
    sc_pt = temp_sum
  
    return
end 


!END OF CAL_SC_GETNN_SURF ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_SC_FIT ROUTINE
!Random rotation walk cost function minimization - fitting to SC 

SUBROUTINE CAL_SC_FIT(con_scmax,con_pamax,con_nnmax,cell,sc_res,sc_ct,   &
    sc_c,sc_pt,sc_p,angle,tsin,tcos,score) 
    
    use KINDS, only: dp    
    IMPLICIT NONE
    
    !Passed 
    integer :: con_scmax
    integer :: con_pamax
    integer :: con_nnmax
    integer :: cell
    integer :: sc_res
    integer :: sc_ct(con_scmax) 
    real(dp)    sc_c(con_scmax,con_pamax,3) 
    integer :: sc_pt
    real(dp) sc_p(con_nnmax,3)
    real(dp) angle(3) 
    real(dp) tsin(360),tcos(360)
    real(dp) score
    
    !Local                                        
    integer :: i
    integer :: xindex, yindex, zindex
    real(dp) drsum, drsum_min, drsum_min_global 
    real(dp) xti(con_scmax), yti(con_scmax), zti(con_scmax)               !initial SC particle positions
    real(dp) xtf(con_scmax), ytf(con_scmax), ztf(con_scmax)               !final SC particle positon after rotations
    integer :: x_ch, y_ch, z_ch
    integer :: xnew, ynew, znew, xold, yold, zold
    integer :: xlow, ylow, zlow
    integer :: sample
    integer :: stuck_counter
    
    

    !Copy SC to temp array      
    do i=1, sc_ct(cell)
        xti(i) = sc_c(cell,i,1)
        yti(i) = sc_c(cell,i,2)
        zti(i) = sc_c(cell,i,3)
    end do
    
    !Initialize global minimum to high value
    drsum_min_global = 1000.0      
    
    !Axial rotational starting locations for random walk          
    do xindex=1, 360, sc_res
        do yindex=1, 360, sc_res
            do zindex=1, 360, sc_res
                
                !Grid value angles and rotated SC coordinates       
                xold = xindex
                yold = yindex
                zold = zindex
                drsum_min = 10000000000000.0   !large initial cost 
                
                
                do sample=1, 100000
                    
                    !Produces random integers -1, 0 or 1 for random walk
                    x_ch = -1 + FLOOR(3*(ran())) 
                    y_ch = -1 + FLOOR(3*(ran())) 
                    z_ch = -1 + FLOOR(3*(ran()))
                    
                    !New attempted rotation point with circle boundary condition
                    xnew = xold + x_ch
                    if(xnew<=0) xnew = xnew + 360
                    if(xnew>360) xnew = xnew - 360
                    
                    ynew = yold + y_ch
                    if(ynew<=0) ynew = ynew + 360
                    if(ynew>360) ynew = ynew - 360
                    
                    znew = zold + z_ch
                    if(znew<=0) znew = znew + 360
                    if(znew>360) znew = znew - 360   
                    
                    !Orient SC to new positions (rotations along x,y,z axis)             
                    CALL CAL_SC_FIT_ROTATE(con_scmax,cell,sc_ct,xti,yti,zti,xtf,ytf,ztf,tsin,tcos,xnew,ynew,znew)
                    
                    
                    !Cost value of new position            
                    CALL CAL_SC_FIT_COST(con_scmax,con_nnmax,cell,drsum,sc_ct,sc_pt,xtf,ytf,ztf,sc_p)  
                    
                    
                    !If cost value decreases, accept new position (gradient descent)               
                    if(drsum<drsum_min) then
                        drsum_min = drsum                               !random walk minimal cost
                        xold = xnew                                     !x angle of minimal cost
                        yold = ynew
                        zold = znew
                        stuck_counter = 0                               !reset counter that count when move is rejected
                    end if
                    
                    !Count of failed moves from position              
                    stuck_counter = stuck_counter + 1   
                    
                    !Exit random walk if stuck (bottom of cost function well)              
                    if(stuck_counter==100) then
                        EXIT
                    end if
                    
                end do
                
                
                !Compare cost for grid point walk against other points for global low            
                if(drsum_min<drsum_min_global) then
                    drsum_min_global = drsum_min
                    xlow = xold
                    ylow = yold
                    zlow = zold       
                end if
                
            end do
            
        end do
        
    end do
    
    angle(1) = xlow
    angle(2) = ylow
    angle(3) = zlow
    score    = drsum_min_global
    
    
    return
end 


!END OF CAL_SC_FIT ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_SC_FIT_COST ROUTINE

SUBROUTINE CAL_SC_FIT_ROTATE(con_scmax,cell,sc_ct,xti,yti,zti,xtf,ytf,ztf,tsin,tcos,xnew,ynew,znew)
    
    use KINDS, only: dp
    IMPLICIT NONE
    
    
    !Passed 
    integer :: con_scmax
    integer :: cell
    integer :: sc_ct(con_scmax) 
    real(dp) xti(con_scmax), yti(con_scmax), zti(con_scmax)
    real(dp) xt2(con_scmax), yt2(con_scmax), zt2(con_scmax)
    real(dp) xt3(con_scmax), yt3(con_scmax), zt3(con_scmax)
    real(dp) xtf(con_scmax), ytf(con_scmax), ztf(con_scmax)
    real(dp) tsin(360),tcos(360)
    integer :: xnew, ynew, znew
    
    !Local     
    integer :: i
    
    
    !Orient SC to new positions (rotations along x,y,z axis)             
    do i=1, sc_ct(cell)
        xt2(i) = xti(i)
        yt2(i) = yti(i)*tcos(xnew) - zti(i)*tsin(xnew)
        zt2(i) = yti(i)*tsin(xnew) + zti(i)*tcos(xnew)          
    end do
    
    do i=1, sc_ct(cell)
        xt3(i) = xt2(i)*tcos(ynew) + zt2(i)*tsin(ynew)
        yt3(i) = yt2(i)
        zt3(i) = -xt2(i)*tsin(ynew) + zt2(i)*tcos(ynew) 
    end do 
    
    do i=1, sc_ct(cell)
        xtf(i) = xt3(i)*tcos(znew) - yt3(i)*tsin(znew)
        ytf(i) = xt3(i)*tsin(znew) + yt3(i)*tcos(znew)
        ztf(i) = zt3(i)
    end do            
    
    return
end 


!END OF CAL_SC_FIT_COST ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_SC_FIT_COST ROUTINE

SUBROUTINE CAL_SC_FIT_COST(con_scmax,con_nnmax,cell,drsum,sc_ct,sc_pt,xtf,ytf,ztf,sc_p)
    
    use KINDS, only: dp  
    IMPLICIT NONE
    
    !Passed 
    integer :: con_scmax
    integer :: con_nnmax
    integer :: cell
    real(dp) ::drsum
    integer :: sc_ct(con_scmax)
    integer :: sc_pt
    real(dp) :: xtf(con_scmax), ytf(con_scmax), ztf(con_scmax)
    real(dp) :: sc_p(con_nnmax,3)          
    
    !Local                  
    integer :: i, j 
    real(dp) :: dx, dy, dz, dr, drmin
    
    
    !Calculate separations value         
    drsum = 0.0            
    do i=1, sc_ct(cell)                                                 !loop over SC particles              
        drmin = 1000000.0
        do j=1, sc_pt                                                   !loop over system particles
            dx = xtf(i) - sc_p(j,1)
            dy = ytf(i) - sc_p(j,2)
            dz = ztf(i) - sc_p(j,3)            
            dr = (dx*dx+dy*dy+dz*dz)  
            
            
            !Find closest separations (squared) btw particles and SC particle
            if(dr<drmin) drmin = dr
        end do
        
        !Sum up all separations values  
        drsum = drsum + sqrt(drmin) 
    end do
    
    !Avg. separation per particle
    drsum = drsum / sc_pt  
    
    return
end 


!END OF CAL_SC_FIT_COST ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_SC_XYZ ROUTINE

subroutine CAL_SC_XYZ(con_scmax,sc_classify,sc_score,sc_orient,tsin,tcos)
    
    use VARIABLES    
    IMPLICIT NONE
    
    !Passed     
    integer :: con_scmax
    integer :: sc_classify(100000)                                       !sc type for each atom
    real(dp)    sc_score(natoms(frame),con_scmax) 
    real(dp)    sc_orient(natoms(frame),con_scmax,3)
    real(dp)    tsin(360),tcos(360)
    
    !Local     
    integer :: i,j
    integer :: cell
    integer :: tempcount(con_scmax)  
    real(dp) x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4
    integer :: xnew,ynew,znew
    
    !Make labels based on coordination
    tempcount = 0
    do i=1, natoms(frame)      
        if(sc_classify(i)/=0) then
            tempcount(sc_classify(i)) = tempcount(sc_classify(i)) + 1
        end if  
    end do

    !Write out classification counts for SC types
    if(headers_done==0) then
        write(70,10)
        10    format('Signature cells classification histogram') 
        write(70,20)
        20    format('   Frame',$)     
        do i=1, in_sc_cells
            cell=in_sc_labs(i)
            write(70,30) cell 
            30      format(i8,$)                    
        end do
    end if
    
    write(70,*) !new line
    
    write(70,40) frame
    40  format(i8,$)    
    do i=1, in_sc_cells             
        cell=in_sc_labs(i)
        write(70,50) tempcount(cell)
        50    format(i8,$)    
    end do
    
    !Write out featureset file
    do i=1, in_sc_cells             
        cell=in_sc_labs(i)
        write(90,51) tempcount(cell)
        51    format(',',i11,$)    
        
        print*,'X:',tempcount(cell)
    end do            
    
    
    !Output xyz file with costs for each fitted SC
    write(72,'(i10,/)') natoms(frame)
    do i=1, natoms(frame)
        
        write(72,100) lc(i),xc(i),yc(i),zc(i)
        100   format(a,f10.5,f10.5,f10.5,$)      
        
        do j=1, in_sc_cells
            cell=in_sc_labs(j) 
            
            write(72,110) sc_score(i,cell)
            110    format(f10.5,$)     
        end do
        
        write(72,*)                                                                                         
    end do
    

    !Write out xyz file
    write(71,'(i10,/)') natoms(frame)
    do i=1, natoms(frame)
        
        !      if(sc_classify(i)/=0) then
        !        x1 = 0.0
        !        y1 = 0.0
        !        z1 = 1.0
        !      
        !        xnew = sc_orient(i,sc_classify(i),1)
        !        ynew = sc_orient(i,sc_classify(i),2)
        !        znew = sc_orient(i,sc_classify(i),3)
        !      
        !        x2 = x1
        !        y2 = y1*tcos(xnew) - z1*tsin(xnew)
        !        z2 = y1*tsin(xnew) + z1*tcos(xnew)          
        !
        !        x3 = x2*tcos(ynew) + z2*tsin(ynew)
        !        y3 = y2
        !        z3 = -x2*tsin(ynew) + z2*tcos(ynew) 
        !
        !        x4 = x3*tcos(znew) - y3*tsin(znew)
        !        y4 = x3*tsin(znew) + y3*tcos(znew)
        !        z4 = z3
        !      end if
        !      
        !      
        !      if(sc_classify(i)==0) then
        !        x4 = 0.0
        !        y4 = 0.0
        !        z4 = 0.0
        !      end if
        
        
        write(71,'(a,2x,3(F10.5,2X),i6)') lc(i),xc(i),yc(i),zc(i),sc_classify(i)  
    end do
    
    
    return
end 


!     END OF CAL_SC_XYZ ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_SU_MAIN ROUTINE

subroutine CAL_SU
    
    use KINDS, only: dp
    use VARIABLES, only: c_lab, frame, headers_done, in_nnhood, lc,    &
    natoms, natoms_max, ncord, nnlist, typecount, xc, yc, zc
    IMPLICIT NONE
 
    
    !Local      
    integer :: i, j, k, l       
    integer :: i1, i2, i3, i4, i5
    integer :: k1, k2, k3, k4      
    integer :: jlabel
    integer :: sum1, sum2, sum3                                
    integer, allocatable :: c_lab_hood(:)                               !particle label for tracking 1st nn environment
    integer, allocatable :: nnhood(:,:,:,:)                             !nnhood(q,r,s,t) the number of particles of type q having r neighbour of type A, s neighbors of type B and t neighbors of type C                
    real(dp) :: p1
    
    
    !Allocate arrays
    ALLOCATE(c_lab_hood(natoms_max))  
    ALLOCATE(nnhood(3,50,50,50))
    

    !Calculate structural units 
    
    nnhood = 0   !Zero array 
    c_lab_hood = 0
    
    do i=1, natoms(frame)
        i1 = c_lab(i)
        sum1 = 0
        sum2 = 0
        sum3 = 0
        do j=1, ncord(i)
            jlabel = nnlist(i,j)
            
            i2 = c_lab(jlabel)
            
            if(i2==1) sum1 = sum1 + 1
            if(i2==2) sum2 = sum2 + 1
            if(i2==3) sum3 = sum3 + 1
        end do
        
        !Add +1 to sums because f90 arrays start at unity
        sum1 = sum1 + 1
        sum2 = sum2 + 1
        sum3 = sum3 + 1
        
        nnhood(i1,sum1,sum2,sum3)=nnhood(i1,sum1,sum2,sum3) + 1   
        
        !Label particle if it has tracked nn enviro.
        do k=1, 10
            
            k1 = in_nnhood(k,1) 
            k2 = in_nnhood(k,2) + 1
            k3 = in_nnhood(k,3) + 1
            k4 = in_nnhood(k,4) + 1
            
            if(k1==i1.and.k2==sum1.and.k3==sum2.and.k4==sum3) then
                c_lab_hood(i) = k
            end if
            
            
        end do
        
    end do   
    
    
    
    
    !All SUs in frame
    if(headers_done==0) then
        write(40,110)
        110   format('STRUCTURAL UNIT POPULATIONS (No. of SUs of central &
        particle type having certain number of type1 type2 and type3 1st nearest neighbors')
        
        write(40,120)
        120   format('      Frame',',       Type',',  No. Type1',',  No. Type2',',  No. Type3', &
                     ',Units Found',',    % Found') 
    end if
    
    do i=1, 3
        do j=1,50
            do k=1, 50
                do l=1, 50
                    if(nnhood(i,j,k,l)/=0) then
                        i1 = nnhood(i,j,k,l)
                        p1 = (real(i1) / real(natoms(frame))) * 100.0_dp
                        if(i1/=0) then
                            write(40,140) frame, i,j-1,k-1,l-1, i1, p1
                            140  format(i11,',',i11,',',i11,',',i11,',',i11,',',i11,',',f11.6)
                        end if
                    end if
                end do
            end do
        end do
    end do
    
    
    
    !Tracked SUs in frame
    if(headers_done==0) then
        write(41,210)
        210   format('STRUCTURAL UNIT POPULATIONS TRACKED')  
        write(41,220)
        220   format('      Frame',',          1',',          2',      &
        ',          3',',          4',',          5',',          6',   &
        ',          7',',          8',',          9',',          10')
     end if
    
    write(41,230) frame
    230 format(i11,$)    
    
    do i=1, 10
        i1 = in_nnhood(i,1)
        i2 = in_nnhood(i,2)
        i3 = in_nnhood(i,3)
        i4 = in_nnhood(i,4)
        
        i5 = nnhood(i1,i2+1,i3+1,i4+1) !+1 for array starting at 0 issue
        
        write(41,240) i5
        240   format(',',i11,$)     
        
    end do
    write(41,*)
    
    
    
    !Write out xyz     
    write(42,'(i10,/)') natoms(frame)       
    do i=1, natoms(frame)
        write(42,'(a,2x,3(F10.5,2X),i6)') lc(i),xc(i),yc(i),zc(i),c_lab_hood(i) 
    end do          
    
    
    !Deallocate
    DEALLOCATE(c_lab_hood)
    DEALLOCATE(nnhood)
    
    
    
    return
end 


!END OF CAL_SU_MAIN ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_RINGS ROUTINE

subroutine CAL_RINGS
    
    use VARIABLES    
    IMPLICIT NONE
    
    integer :: ringsize
    integer :: i, j, k, l, loop_nn
    integer :: i1, i2
    integer :: n1, n2
    integer :: count_now, count_next
    integer :: count1, count2
    integer :: list_now(natoms_max), list_next(natoms_max)
    integer :: dis_to_centre(natoms_max)    
    
    integer :: store1(1000,10)
    integer :: store2(1000,10)
    
    integer :: paths
    integer :: path_length(1000)
    integer :: path_list(1000,10)
    
    
    
    
    !     Pick particle i (vertex in network)
    !     Loop out to maximum atom neighbors in a looping sequence over 1st nn, then nn of 1st nn, etc
    !     Labelling uinmodally (starting atom 0, 1st neighbor = 1, 2nd neighbor = 2)
    !     If path encounters a labelled atom, a ring is found
    !     Store ring sequences
    !     Check for SP rings (see if smaller rings exist within larger ones)
    !
    !     Path sequences (atomic labels) stored in list_path(i,j) where i is the path label and j is the jth atom in the path
    !     path_length(i) stores the length of the path (atoms) of the ith path    
    
    
    
    !Loop out over candidates and label using unimodal labelling 
    dis_to_centre = -1          !start all distances as negative      
    
    
    
    !       -initialize at atom 1 (FIX for loop over N later)
    count_now = 1      
    list_now(1) = 1 
    dis_to_centre(1) = 0
    path_length = 0
    path_list = 0
    paths = 0
    
    
    !       -path initialization
    
    
    do loop_nn = 1, 5                                               !ith nearest neighbor loop
        
        print*,''
        print*,loop_nn,'NN LOOP'
        
        count_next = 0
        do i=1, count_now                                               !loop over ith nn atoms
            i1 = list_now(i)
            
            do j=1, ncord(i1)                                             !loop over ith + 1 nn atoms 
                i2 = nnlist(i1,j)
                
                !             -Even ring found                
                if(dis_to_centre(i2)>dis_to_centre(i1)) then            
                    ringsize = loop_nn * 2
                    print*,i1,i2,dis_to_centre(i2),ringsize,'ring found'
                end if
                
                
                !             -Odd ring found            
                if(dis_to_centre(i2)==dis_to_centre(i1)) then            
                    ringsize = loop_nn * 2 - 1
                    print*,i1,i2,dis_to_centre(i2),ringsize,'ring found'
                end if
                
                
                !             -Store unlabelled atoms            
                if(dis_to_centre(i2)==-1) then  
                    
                    print*,i1,i2,dis_to_centre(i2),'UNLABELLED!'  
                    
                    count_next = count_next + 1
                    list_next(count_next) = i2
                    dis_to_centre(i2) = loop_nn                               !atoms labelled unimodelly  
                    
                end if            
                
                !             -Track path
                if(loop_nn==1) then
                    paths = paths + 1
                    path_list(paths,loop_nn) = i2  
                end if
                
                if(loop_nn==2) then
                    do k=1,paths
                        if(path_list(k,1)==i1) then
                            print*,'P:',k,path_list(k,1), i1,i2
                        end if
                    end do
                    
                end if
                
                
                
                
                
            end do
            
        end do
        
        
        
        
        
        !         -Copy list of next nn for upcoming iteration
        count_now = count_next
        do i=1, count_next 
            list_now(i) = list_next(i) 
        end do
        
    end do
    
    do i=1, paths
        print*,'i, P:',i,path_list(i,1), path_list(i,2), path_list(i,3), path_list(i,4),path_list(i,5)   
        
    end do
    
    
    
    
    return
end 


!END OF CAL_RINGS ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_Q6 ROUTINE
!q6 order based analysis

!Based on the work of Dr Brendan O'Malley from his PhD thesis 'Molecular
!Dynamics Investigation of Crystallization in the Hard Sphere System'

subroutine CAL_Q6
   
    use KINDS, only: dp
    use VARIABLES, only: con_maxnn, natoms_max      
    IMPLICIT NONE


    real(dp) :: atom_phi(natoms_max,con_maxnn)                          !phi angle between atom i and j  
    real(dp) :: atom_theta(natoms_max,con_maxnn)                        !theta angle between atom i and j  
    integer :: q6connect(natoms_max)                                    !q6.q6 similarity coordination of atom i    
    real(dp) :: q6q6_val(natoms_max)
    complex :: q6i(natoms_max,-6:6)  
    
    
    call CAL_Q6_ANGLES(atom_phi,atom_theta)
    call CAL_Q6_BONDS(atom_phi,atom_theta,q6i)
    call CAL_Q6_ORDER(q6i,q6connect,q6q6_val)
    call CAL_Q6_Q6Q6HIST(q6connect)
    call CAL_Q6_XYZ(q6connect,q6q6_val)
    
    
    return
end 

!END OF CAL_Q6 ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_Q6_ANGLES ROUTINE

subroutine CAL_Q6_ANGLES(atom_phi,atom_theta)
    
    use KINDS, only: dp
    use VARIABLES, only: con_maxnn, con_pi, frame, natoms, natoms_max, &
    ncord, nnlist, xc, yc, zc, xl, yl, zl, xl2inv, yl2inv, zl2inv 
    IMPLICIT NONE

    !Passed
    real(dp) :: atom_phi(natoms_max,con_maxnn)    
    real(dp) :: atom_theta(natoms_max,con_maxnn)

    !Local
    integer :: i, j, fbond
    real(dp) :: tol
    real(dp) :: xci, yci, zci
    real(dp) :: xuf, yuf, zuf, ruf
    real(dp) :: rdis, cosphi, cos_theta
    

    !Set tolerance for acos function
    tol = 0.00001_dp
    
    do i=1, natoms(frame)
        xci = xc(i)
        yci = yc(i)       
        zci = zc(i)
        
        do j=1,ncord(i)
            if(ncord(i)/=0) then
                fbond = nnlist(i,j)
                xuf = xc(fbond) - xci
                yuf = yc(fbond) - yci
                zuf = zc(fbond) - zci
                
                xuf = xuf - (int(xuf*xl2inv)*xl)
                yuf = yuf - (int(yuf*yl2inv)*yl)
                zuf = zuf - (int(zuf*zl2inv)*zl)
                
                ruf=xuf*xuf+yuf*yuf+zuf*zuf
                
                !Calculate phi
                rdis = dsqrt(xuf**2 + yuf**2)
                
                if(rdis/=0.0_dp) then
                    cosphi = xuf/rdis
                    if(abs(1.0_dp+cosphi)<=tol) then
                    cosphi = -0.999999_dp
                    else if(abs(-1+cosphi)<=tol) then
                        cosphi = 0.9999999_dp
                    endif
                    atom_phi(i,j) = dacos(cosphi)
                    
                    !Check quadrant
                    if(yuf<0.0_dp) then 
                        atom_phi(i,j) = 2.0_dp*con_pi - atom_phi(i,j)
                    end if
                
                else
                    atom_phi(i,j) = 0.0_dp
                end if
                
                !Check orientation if bond along x-axis
                if(yuf==0.0_dp) then
                    if(xuf>0.0_dp) then
                    atom_phi(i,j) = 0.0_dp
                    else
                        atom_phi(i,j) = con_pi
                    end if
                end if
                
                
                !Calculate theta
                cos_theta = zuf / sqrt(ruf)
                atom_theta(i,j) = dacos(cos_theta)
                
            end if
        end do
    end do
    

    
    return
end 


!END OF CAL_Q6_ANGLES ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_Q6_BONDS ROUTINE

subroutine CAL_Q6_BONDS(atom_phi,atom_theta,q6i)
    
    use KINDS, only: dp
    use VARIABLES, only: con_maxnn, con_pi, frame, natoms, natoms_max, &
    ncord
    
    
    IMPLICIT NONE

    !Passed
    real(dp) :: atom_phi(natoms_max,con_maxnn) 
    real(dp) :: atom_theta(natoms_max,con_maxnn)
    complex :: q6i(natoms_max,-6:6)
    
    !Local
    integer :: i, j, il, im, imm
    integer :: total_bonds_all 
    integer :: lmax
    integer :: npts_q6dot  
    integer :: npts_q6    
    real(dp) :: del_q6  
    real(dp) :: del_q6dot    
    real(dp) :: qlsum2(40), qlsum2b(40)
    real(dp) :: qlmreal, qlmreal2, qlmimg, qlmimg2, qlm2
    real(dp) :: total_bonds
    real(dp) :: phi, theta
    real(dp) :: ql(40), qlbond(40)    
    complex :: qlm, qlmsum, qlmav 
    complex :: qlmsum_bond, qlmav_bond
    complex :: CAL_Q6_YLM
    complex :: qlmsingle
    complex :: cqlmav(20,40)    
    

    
    !Set constants for Spherical Harmonics
    lmax = 12   
    del_q6dot = 0.01
    npts_q6dot = 1.0 / del_q6dot
    del_q6 = 0.01
    npts_q6 = 2.0 / del_q6 + 1
    
    
    
    !Zero variables
    qlm = (0,0)
    qlmav = (0,0)
    qlmav_bond = (0,0)
    
    do i=1, 40
        qlsum2(i) = 0.d0
        qlsum2b(i) = 0.d0
        ql(i)= 0.d0
        qlbond(i) = 0.d0
    end do
    
    do i=1, 20
        do j=1, 40
            cqlmav(i,j)=(0,0)
        enddo
    enddo
    
    qlmreal  = 0.d0
    qlmreal2 = 0.d0
    qlmimg   = 0.d0
    qlmimg2  = 0.d0
    qlm2     = 0.d0
    
    
    
    do il=0, lmax, 1
        do im = -il, il
            qlmsum = (0,0)
            qlmsum_bond = (0,0)
            total_bonds = 0
            
            !Calculate qlm for all atoms
            do i=1, natoms(frame)
                total_bonds = total_bonds + ncord(i)
                
                !zero qlm for a single atom
                qlmsingle = (0,0)
                
                !loop over neighbors of atom i
                do j=1, ncord(i)
                    theta = atom_theta(i,j)
                    phi   = atom_phi(i,j)
                    
                    
                    !Calculate qlm for a given l, m and bond ij
                    if(im>=0) then
                    qlm = CAL_Q6_YLM(il,im,theta,phi,con_pi)
                    else
                        imm = -im
                        qlm = CAL_Q6_YLM(il,imm,theta,phi,con_pi)
                        qlm = ((-1)**imm)*conjg(qlm)
                    endif
                    
                    
                    !Sum up single contribution
                    qlmsingle = qlmsingle + qlm
                enddo
                
                if(ncord(i)/=0) then
                    qlmsum = qlmsum + qlmsingle/ncord(i)
                endif
                qlmsum_bond = qlmsum_bond + qlmsingle
                
                !Store m components for l = 6
                if(il==6) then
                    if(ncord(i)/=0) then
                        q6i(i,im) = qlmsingle/ncord(i)
                    endif              
                endif
                
            end do
            
            !Calculate the different averages
            
            qlmav = qlmsum/natoms(frame)
            qlmav_bond = qlmsum_bond/total_bonds
            
            !Sum up rotationally averaged Ql
            if(im<0) then
            imm=-im+il
            else
                imm=im
            endif
            
            !Average over atoms
            cqlmav(il+1,imm+1) = qlmav
            qlmreal  = real(cqlmav(il+1,imm+1))
            qlmreal2 = qlmreal**2
            qlmimg   = aimag(cqlmav(il+1,imm+1))
            qlmimg2  = qlmimg**2
            qlm2  = qlmreal2+qlmimg2
            qlsum2(il+1) = qlsum2(il+1)+qlm2
            
            !Average over bonds
            cqlmav(il+1,imm+1) = qlmav_bond
            qlmreal  = real(cqlmav(il+1,imm+1))
            qlmreal2 = qlmreal**2
            qlmimg   = aimag(cqlmav(il+1,imm+1))
            qlmimg2  = qlmimg**2
            qlm2  = qlmreal2+qlmimg2
            qlsum2b(il+1) = qlsum2b(il+1) + qlm2
        enddo
        ql(il+1)=sqrt((4*con_pi/(2*il+1))*qlsum2(il+1))
        qlbond(il+1) = sqrt((4*con_pi/(2*il+1))*qlsum2b(il+1))
    enddo
    total_bonds_all = total_bonds_all + total_bonds
    
    
    
    return
end 


!END OF CAL_Q6_BONDS ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_Q6_ORDER ROUTINE
!Purpose: To compute the set of invariants which are a
!         local measure of crystallinity.
!Method:  Define for particle a normalised 2l+1 vector
!         where l=6 i.e. 2l+1=13.
!         See Eqn's 2.7-2.11 in Brendan O'Malley's Ph.D. Thesis
!Reference: J. S. van Duijeveldt and D. Frenkel,
!           Computer simulation study of free energy
!           barriers in crystal nucleation",
!           J. Chem. Phys. 96(6), 15 March 1992

subroutine CAL_Q6_ORDER(q6i, q6connect, q6q6_val)
    
    use KINDS, only: dp
    use VARIABLES, only: con_pi, frame, natoms_max, natoms, ncord,     &
    nnlist, in_q6order_dotmin     
    IMPLICIT NONE

    !Passed
    complex :: q6i(natoms_max,-6:6)
    integer :: q6connect(natoms_max)                                    
    real(dp) :: q6q6_val(natoms_max)
    
    !Local
    integer :: i, j, k, m
    real(dp) :: qi(natoms_max)
    real(dp) :: q6m_sumsq
    real(dp) :: sum_dot_product, dot_product_m
    
    
    q6connect = 0
    q6q6_val = 0
    
    do i=1, natoms(frame)
        qi(i) = 0.d0
    end do
    
    
    do i=1, natoms(frame)
        
        !Calculate the q6(i) and premultiple by factor
        q6m_sumsq = 0.d0
        do m=-6, 6
            q6m_sumsq=q6m_sumsq+(real(q6i(i,m)))**2 + (aimag(q6i(i,m)))**2
        end do
        qi(i)=sqrt((4*con_pi/(13))*q6m_sumsq)
        
        !Normalise the q6mi
        q6m_sumsq = sqrt(q6m_sumsq)
        do m=-6, 6
            if(q6m_sumsq/=0.d0) then
                q6i(i,m) = q6i(i,m)/q6m_sumsq
            end if
        end do
        
    end do
    
    
    
    !Calculate the dot product
    do i=1, natoms(frame)
        do k=1, ncord(i)
            sum_dot_product = 0.d0
            j = nnlist(i,k)
            do m=-6, 6
                dot_product_m = real(q6i(i,m))*real(q6i(j,m))              &
                + aimag(q6i(i,m))*aimag(q6i(j,m))
                sum_dot_product = sum_dot_product + dot_product_m
            end do
            
            !-----------------------------------------------------          
            !This is the van Duijneveldt and Frenkel criterion for
            !determining whether the two particles are connected
            !where the dot product is greater than the preset 0.5
            !(that is, their spherical harmonics are in phase)
            !Here the preset is an input variable in_q6order_dotmin.
            !-----------------------------------------------------
            if(sum_dot_product>in_q6order_dotmin) then
                q6connect(i) = q6connect(i) + 1
            end if
            
        end do
        q6q6_val(i) = sum_dot_product
    end do
    

    return
end 


!END OF CAL_Q6_ORDER ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_Q6_Q6Q6HIST ROUTINE

subroutine CAL_Q6_Q6Q6HIST(q6connect)
    
    use KINDS, only: dp
    use VARIABLES, only: flag_surf, frame, headers_done, natoms,       &
    natoms_max     
    IMPLICIT NONE
    
    !Passed
    integer :: q6connect(natoms_max)        

    !Local
    integer :: i
    integer :: counter
    integer :: q6connect_distbn(22)
    integer :: q6connect_distbnS(22)
    integer :: q6connect_distbnB(22) 
    real(dp) :: q6connect_avg, q6connect_avgB, q6connect_avgS      
    
    
    !Zero quantities
    q6connect_distbn = 0
    q6connect_distbnB = 0
    q6connect_distbnS = 0
    q6connect_avg = 0
    q6connect_avgB = 0
    q6connect_avgS = 0
    
    
    !q6.q6 histogram
    
    !total histogram
    counter = 0
    do i=1, natoms(frame)
        if(q6connect(i)>0.and.q6connect(i)<=20) then
            q6connect_distbn(q6connect(i))=q6connect_distbn(q6connect(i))+1
        end if
        if(q6connect(i)>20) then
            q6connect_distbn(21) = q6connect_distbn(21) + 1
        end if
        if(q6connect(i)==0) then
            q6connect_distbn(22) = q6connect_distbn(22) + 1
        end if
        
        q6connect_avg = q6connect_avg + q6connect(i)
        counter = counter + 1
    end do
    if(counter>0) then
    q6connect_avg = q6connect_avg / counter
    else
        q6connect_avg = 0.0
    end if
    
    
    !bulk histogram
    counter = 0
    do i=1, natoms(frame)
        if(flag_surf(i)==0) then
            if(q6connect(i)>0.and.q6connect(i)<=20) then
                q6connect_distbnB(q6connect(i))=q6connect_distbnB(q6connect(i))+1
            end if
            if(q6connect(i)>20) then
                q6connect_distbnB(21) = q6connect_distbnB(21) + 1
            end if
            if(q6connect(i)==0) then
                q6connect_distbnB(22) = q6connect_distbnB(22) + 1
            end if
            
            q6connect_avgB = q6connect_avgB + q6connect(i)
            counter = counter + 1
        end if  
    end do
    if(counter>0) then
    q6connect_avgB = q6connect_avgB / counter
    else
        q6connect_avgB = 0.0
    end if   
    
    
    !surf histogram
    counter = 0
    do i=1, natoms(frame)
        if(flag_surf(i)==1) then
            if(q6connect(i)>0.and.q6connect(i)<=20) then
                q6connect_distbnS(q6connect(i))=q6connect_distbnS(q6connect(i))+1
            end if
            if(q6connect(i)>20) then
                q6connect_distbnS(21) = q6connect_distbnS(21) + 1
            end if
            if(q6connect(i)==0) then
                q6connect_distbnS(22) = q6connect_distbnS(22) + 1
            end if
            
            q6connect_avgS = q6connect_avgS + q6connect(i)
            counter = counter + 1
        end if  
    end do
    if(counter>0) then
    q6connect_avgS = q6connect_avgS / counter
    else
        q6connect_avgS = 0.0
    end if       
    
    
    
    
    !Write out histograms      
    if(headers_done==0) then
        write(60,110)
        110 format('Q6Q6 BOND COORDINATION')  
        write(60,120)
        120 format('      Frame',$)
        write(60,130)
        130 format(',  Avg Total',',   Avg Bulk',',   Avg Surf'           &
        ,',         T0',',         T1',',         T2',',         T3',',         T4' &
        ,',         T5',',         T6',',         T7',',         T8',',         T9' &
        ,',        T10',',        T11',',        T12',',        T13',',        T14' &
        ,',        T15',',        T16',',        T17',',        T18',',        T19' &
        ,',        T20',',       T>20'                                        &
        ,',         B0',',         B1',',         B2',',         B3',',         B4' &
        ,',         B5',',         B6',',         B7',',         B8',',         B9' &
        ,',        B10',',        B11',',        B12',',        B13',',        B14' &
        ,',        B15',',        B16',',        B17',',        B18',',        B19' &
        ,',        B20',',       B>20'                                        &
        ,',         S0',',         S1',',         S2',',         S3',',         S4' &
        ,',         S5',',         S6',',         S7',',         S8',',         S9' &
        ,',        S10',',        S11',',        S12',',        S13',',        S14' &
        ,',        S15',',        S16',',        S17',',        S18',',        S19' &
        ,',        S20',',       S>20')  
        
    end if
    
    !output averages
    write(60,140) frame,q6connect_avg,q6connect_avgB,q6connect_avgS                                          
    140 format(i11,3(',',f11.6),$)   
    
    !write out feature file   
    write(90,141) q6connect_avg,q6connect_avgB,q6connect_avgS                                 
    141 format(',',f11.6,',',f11.6,',',f11.6,$)  
    
    
    
    !output total coordination histogram  
    write(60,150) q6connect_distbn(22)
    150 format(',',i11,$)
    
    !write out feature file
    write(90,151) q6connect_distbn(22)
    151 format(',',i11,$)    
    
    do i=1, 21
        write(60,160) q6connect_distbn(i)
        160   format(',',i11,$)      
    end do
    
    !write out feature file
    do i=1, 21
        write(90,161) q6connect_distbn(i)
        161   format(',',i11,$)      
    end do
    
    
    
    
    !output bulk coordination histogram      
    write(60,170) q6connect_distbnB(22)
    170 format(',',i11,$)
    
    !write out feature file
    write(90,171) q6connect_distbnB(22)
    171 format(',',i11,$)  
    
    do i=1, 21
        write(60,180) q6connect_distbnB(i)
        180   format(',',i11,$)      
    end do
    
    !write out feature file      
    do i=1, 21
        write(90,181) q6connect_distbnB(i)
        181   format(',',i11,$)      
    end do
    
    
    
    
    !output surface coordination histogram      
    write(60,190) q6connect_distbnS(22)
    190 format(',',i11,$)
    
    !write out feature file  
    write(90,191) q6connect_distbnS(22)
    191 format(',',i11,$)
    
    do i=1, 21
        write(60,200) q6connect_distbnS(i)
        200   format(',',i11,$)      
    end do 
    
    !write out feature file       
    do i=1, 21
        write(90,201) q6connect_distbnS(i)
        201   format(',',i11,$)      
    end do   
    

    write(60,*)
    
    return
end 


!END OF CAL_Q6_Q6Q6HIST ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_Q6_XYZ ROUTINE
!Output q6.q6 similarity coordination for visualization

subroutine CAL_Q6_XYZ(q6connect, q6q6_val)
    
    use KINDS, only: dp
    use VARIABLES, only: frame, lc, natoms, natoms_max, xc, yc, zc, xyz_prec_str
    IMPLICIT NONE
    
    !Passed
    integer :: q6connect(natoms_max)        
    real(dp) :: q6q6_val(natoms_max)
    
    !Local 
    integer :: i
    
    
    !XYZ OUTPUT
    write(61,'(i10,/)') natoms(frame)        
    do i=1, natoms(frame)
        write(61,'(a,2x,3(F10.'//xyz_prec_str//',2X),i3)') lc(i),xc(i),yc(i),zc(i),     &
        q6connect(i) 
    end do
 
    
    return
end       

!END OF CAL_Q6_XYZ ROUTINE
!=======================================================================



!=======================================================================
!START OF CAL_Q6_YLM FUNCTION
!Purpose: To compute the spherical harmonics ylm(theta,phi)
!equal to qlm(r)=ylm(theta(r),phi(r)) where r is the
!position vector for the midpoint of a geometric bond
!Note: the final ylm value is a complex number

complex function CAL_Q6_YLM(l,m,theta,phi,con_pi)
    
    use KINDS
    use KINDS, only: dp
    IMPLICIT NONE
    
    integer :: l, m, diff, sum_int
    real(dp) :: x,ylmre,ylmim,theta,phi
    real(dp) :: const1, const2, argsq, const, xtheta
    real(dp) :: flg,  ylmreal, ylmimag
    real(dp) :: con_pi
    real(dp) :: CAL_Q6_PLGNDR
    real(dp) :: CAL_Q6_FACTR2
    
    const1 = (2*l+1)/(4*con_pi)
    diff   = l-m
    sum_int = l+m
    const2 = CAL_Q6_FACTR2(sum_int,diff)
    argsq  = const1*const2
    const  = sqrt(argsq)
    x = dcos(theta)
    xtheta = sngl(x)
    flg   = CAL_Q6_PLGNDR(l,m,xtheta)
    ylmre = const*flg*dcos(m*phi)
    ylmim = const*flg*dsin(m*phi)
    ylmreal = sngl(ylmre)
    ylmimag = sngl(ylmim)
    CAL_Q6_YLM  = cmplx(ylmreal,ylmimag)
    
    
    
    return
end 


!END OF CAL_Q6_YLM FUNCTION
!=======================================================================



!=======================================================================
!START OF CAL_Q6_FACTR2 FUNCTION
!Purpose: To calculate ((l-m)!/(l+m)!)=diff!/sum!

double precision function CAL_Q6_FACTR2(sums,diff)

    use KINDS, only: dp
    IMPLICIT NONE
    

    integer :: i, sums, diff
    real(dp) :: produc
    
    produc=1
    do i=sums, diff+1, -1
        produc=produc*i
    enddo
    CAL_Q6_FACTR2=1/produc
    
    return
end


!END OF CAL_Q6_FACTR2 FUNCTION
!=======================================================================



!=======================================================================
!START OF CAL_Q6_PLGNDR FUNCTION
!Purpose: To compute the associated legendre polynomial pml(x).
!here m and l are integers satisfying 0<=m=<l while
!x lies in the range -1<=x=<+1

double precision function CAL_Q6_PLGNDR(l,m,x)

    use KINDS
    IMPLICIT NONE

    integer :: l, m, i, ll
    real(dp) :: x, somx2, pll, pmm, fact
    real(dp) ::  pmmp1
    
    if((m<0).or.(m>l).or.(abs(x)>1)) stop 'bad arguments'
    pmm=1
    if(m>0) then
        somx2=sqrt((1.-x)*(1.+x))
        fact=1
        do i = 1, m
            pmm=-pmm*fact*somx2
            fact=fact+2
        enddo
    endif
    if(l==m) then
    CAL_Q6_PLGNDR=pmm
    else
        pmmp1=x*(2*m+1)*pmm
        if(l==m+1) then
        CAL_Q6_PLGNDR=pmmp1
        else
            do ll=m+2,l
                pll=(x*(2*ll-1)*pmmp1-(ll+m-1)*pmm)/(ll-m)
                pmm=pmmp1
                pmmp1=pll
            enddo
            CAL_Q6_PLGNDR=pll
        endif
    endif
    
    return
end


!END OF CAL_Q6_PLGNDR FUNCTION
!=======================================================================


!=======================================================================
!START OF CAL_FRACTAL_ATOM_CENTRE ROUTINE
!Fractal dimension box counting method adapted from:
!Motofumi T. Suzuki, A Three Dimensional Box Counting Method for 
!Measuring Fractal Dimension of 3D Models, The 11th IASTED 
!International Conference on Internet and Multimedia Systems 
!and Applications (IMSA_2007), Hawaii, USA, 08/2007.   
!
!Fractal Dimension = log(Nr) / log (1/r)
!where Nr is the number of occupied grids of length r
!TODO: 
!  - Break out of loop when count is maximum

subroutine CAL_FRACTAL_ATOM_CENTRE
    
    use KINDS, only: dp
    use VARIABLES, only: frame, headers_done, in_fradim_atom_centre_maxres,        &
    min_x, min_y, min_z, natoms, xc, yc, zc, xl, yl, zl   
    IMPLICIT NONE 
    
    !Local      
    integer :: i, j
    integer :: xt, yt, zt
    integer, allocatable :: cell(:,:,:) 
    integer, allocatable :: counter(:) 
    real(dp) :: cell_resx, cell_resy, cell_resz
    
    ALLOCATE(cell(2*(2**in_fradim_atom_centre_maxres),2*(2**in_fradim_atom_centre_maxres),2*(2**in_fradim_atom_centre_maxres)))
    ALLOCATE(counter(in_fradim_atom_centre_maxres*2))
    
    cell = 0
    
    print*,"-Performing box-counting technique on atom centres to estimate fractal dimension..."
    do i=1, in_fradim_atom_centre_maxres*2

        !zero counter           
        counter(i) = 0  
        
        !determine grid lengths        
        cell_resx = xl / (2**(i*0.5))
        cell_resy = yl / (2**(i*0.5))
        cell_resz = zl / (2**(i*0.5))
        
        !bin com locations        
        do j=1, natoms(frame)
            
            !shift particle position to positive values (array needs it)          
            xt = int((xc(j)-min_x)/cell_resx)+1
            yt = int((yc(j)-min_y)/cell_resy)+1
            zt = int((zc(j)-min_z)/cell_resz)+1
            
            
            if(cell(xt,yt,zt)==0) then
                cell(xt,yt,zt) = 1
                counter(i) = counter(i) + 1
            end if   
        end do
        
        !Zero cell array        
        cell = 0
        
    end do
    
    !Write out data
    if(headers_done==0) then
        write(80,10)
        10    format('FRACTAL DIMENSION (2nd line: LOG[1/r]  Below 2nd line: LOG[no of grids occupied])') 
        write(80,20)
        20    format('      Frame',$)        
        do i=1, in_fradim_atom_centre_maxres*2
            write(80,30) log10((2**(i*0.5))/xl)
            30      format(',',f11.6,$)                    
        end do
    end if
    
    write(80,*) !new line
    
    write(80,40) frame
    40  format(i11,$)    
    do i=1, in_fradim_atom_centre_maxres*2
        write(80,50) log10(real(counter(i)))
        50    format(',',f11.6,$)    
    end do

    !Deallocate 
    DEALLOCATE(cell)
    DEALLOCATE(counter)
    
    return
end 

!END OF CAL_FRACTAL_ATOM_CENTRE ROUTINE
!=======================================================================


!=======================================================================
!START OF CAL_FRACTAL_SATOM_BALL ROUTINE
!Fractal dimension box counting method adapted from:
!Motofumi T. Suzuki, A Three Dimensional Box Counting Method for 
!Measuring Fractal Dimension of 3D Models, The 11th IASTED 
!International Conference on Internet and Multimedia Systems 
!and Applications (IMSA_2007), Hawaii, USA, 08/2007.   
!
!Fractal Dimension = log(Nr) / log (1/r)
!where Nr is the number of occupied grids of length r
!TODO:
!  - Use proper atomic radii
!  - Figure out mathematics behind checking if a cell covers the surface

subroutine CAL_FRACTAL_SATOM_BALL
    
    use KINDS, only: dp
    use VARIABLES, only: frame, headers_done, min_x, min_y, min_z, &
    natoms, xc, yc, zc, xl, yl, zl, flag_surf, in_fradim_satom_ball_maxres
    IMPLICIT NONE 
    
    !Local      
    integer :: magn_fac
    integer :: i, j, k, l, m
    integer :: xt, yt, zt
    integer :: atom_cell_idxx, atom_cell_idxy, atom_cell_idxz
    integer :: scan_cell_idxx, scan_cell_idxy, scan_cell_idxz
    integer :: num_scanx, num_scany, num_scanz
    integer, allocatable :: cell(:,:,:) 
    integer, allocatable :: counter_all(:), counter_surf(:), counter_overlap(:), counter_surf_only(:)
    real(dp) :: cell_resx, cell_resy, cell_resz
    real(dp) :: scan_cell_minx, scan_cell_maxx, scan_cell_miny, scan_cell_maxy, scan_cell_minz, scan_cell_maxz
    real(dp) :: scan_cell_nearestx, scan_cell_nearesty, scan_cell_nearestz
    real(dp) :: dist, atom_rad
    
    ALLOCATE(cell(2**in_fradim_satom_ball_maxres,2**in_fradim_satom_ball_maxres,2**in_fradim_satom_ball_maxres))
    ALLOCATE(counter_all(in_fradim_satom_ball_maxres*2))
    ALLOCATE(counter_surf(in_fradim_satom_ball_maxres*2))
    ALLOCATE(counter_overlap(in_fradim_satom_ball_maxres*2))
    ALLOCATE(counter_surf_only(in_fradim_satom_ball_maxres*2))
    
    cell = 0
    atom_rad = 3.5

    print*,"-Performing box-counting technique on surface atom balls to estimate fractal dimension..."
    do i=1, in_fradim_satom_ball_maxres*2

        !zero counters
        counter_all(i) = 0  
        counter_surf(i) = 0  
        counter_overlap(i) = 0  
        counter_surf_only(i) = 0  
        
        !determine grid lengths
        magn_fac = 2**(i*0.5)
        cell_resx = xl / magn_fac
        cell_resy = yl / magn_fac
        cell_resz = zl / magn_fac
        print'(a40,i10,f10.3,f10.3,f10.3)','  -Magnification, grid length (x, y, z): ',magn_fac,cell_resx,cell_resy,cell_resz

        !determine number of scans to do around each surface atom  !Yet to be refined!!
        num_scanx = int(2*atom_rad/cell_resx)
        num_scany = int(2*atom_rad/cell_resy)
        num_scanz = int(2*atom_rad/cell_resz)
        !print*,'   -Number of cells to be scanned around atom: ',num_scanx,num_scany,num_scanz

        do j=1, natoms(frame)

            !identify index of the atom relative to coordinates generated based on current grid length
            atom_cell_idxx = int((xc(j)-min_x)/cell_resx) + 1
            atom_cell_idxy = int((yc(j)-min_y)/cell_resy) + 1
            atom_cell_idxz = int((zc(j)-min_z)/cell_resz) + 1
            !print*,'atom_cell_idxx,atom_cell_idxy,atom_cell_idxz: ',atom_cell_idxx,atom_cell_idxy,atom_cell_idxz

            !scan the cells around the atom
            do k=-num_scanx, num_scanx
                scan_cell_idxx = atom_cell_idxx + k
                if(scan_cell_idxx<1 .or. scan_cell_idxx>magn_fac) cycle
                scan_cell_minx = min_x + scan_cell_idxx*cell_resx - cell_resx
                scan_cell_maxx = min_x + scan_cell_idxx*cell_resx
                if(xc(j)<scan_cell_minx) then
                    scan_cell_nearestx = scan_cell_minx
                else if(xc(j)>scan_cell_maxx) then
                    scan_cell_nearestx = scan_cell_maxx
                else
                    scan_cell_nearestx = xc(j)
                end if
                !print*,scan_cell_minx,scan_cell_maxx,scan_cell_nearestx,xc(j)

                do l=-num_scany, num_scany
                    scan_cell_idxy = atom_cell_idxy + l
                    if(scan_cell_idxy<1 .or. scan_cell_idxy>magn_fac) cycle
                    scan_cell_miny = min_y + scan_cell_idxy*cell_resy - cell_resy
                    scan_cell_maxy = min_y + scan_cell_idxy*cell_resy
                    if(yc(j)<scan_cell_miny) then
                        scan_cell_nearesty = scan_cell_miny
                    else if(yc(j)>scan_cell_maxy) then
                        scan_cell_nearesty = scan_cell_maxy
                    else
                        scan_cell_nearesty = yc(j)
                    end if

                    do m=-num_scanz, num_scanz
                        scan_cell_idxz = atom_cell_idxz + m
                        if(scan_cell_idxz<1 .or. scan_cell_idxz>magn_fac) cycle
                        scan_cell_minz = min_z + scan_cell_idxz*cell_resz - cell_resz
                        scan_cell_maxz = min_z + scan_cell_idxz*cell_resz
                        if(zc(j)<scan_cell_minz) then
                            scan_cell_nearestz = scan_cell_minz
                        else if(zc(j)>scan_cell_maxz) then
                            scan_cell_nearestz = scan_cell_maxz
                        else
                            scan_cell_nearestz = zc(j)
                        end if

                        !check whether the scanned cell belongs to surface/bulk/void
                        dist = sqrt((xc(j)-scan_cell_nearestx)**2 + (yc(j)-scan_cell_nearesty)**2 + (zc(j)-scan_cell_nearestz)**2)
                        !print*,dist,xc(j),scan_cell_nearestx
                        if(dist<atom_rad) then
                            if(flag_surf(j)==0) then  !for bulk atom
                                if(cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz)==0) then  !if unscanned, mark as bulk
                                    cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz) = 1
                                    counter_all(i) = counter_all(i) + 1
                                else if(cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz)==1) then  !skip if already marked as bulk
                                    cycle
                                else if(cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz)==2) then  !if marked as surf, mark as overlap
                                    cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz) = 3
                                    counter_overlap(i) = counter_overlap(i) + 1
                                else if(cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz)==3) then  !skip if already marked as overlap
                                    cycle
                                end if
                            else if(flag_surf(j)==1) then  !for surface atom
                                if(cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz)==0) then  !if unscanned, mark as surf
                                    cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz) = 2
                                    counter_all(i) = counter_all(i) + 1
                                    counter_surf(i) = counter_surf(i) + 1
                                else if(cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz)==1) then  !if marked as bulk, mark as overlap
                                    cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz) = 3
                                    counter_overlap(i) = counter_overlap(i) + 1
                                    counter_surf(i) = counter_surf(i) + 1
                                else if(cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz)==2) then  !skip if already marked as surf
                                    cycle
                                else if(cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz)==3) then  !skip if already marked as overlap
                                    cycle
                                end if
                            end if
                        end if   
                    end do
                end do
            end do
        end do
        
        !Zero cell array        
        cell = 0
        print*,'   -Counts (all, surf, overlap):               ',counter_all(i),counter_surf(i),counter_overlap(i)
        
    end do
    counter_surf_only = counter_surf - counter_overlap
    
    !Write out data
    if(headers_done==0) then
        write(81,10)
        10    format('FRACTAL DIMENSION (2nd line: LOG[1/r]  Below 2nd line: LOG[no of grids occupied])') 
        write(81,20)
        20    format('      Frame',$)        
        do i=1, in_fradim_satom_ball_maxres*2
            write(81,30) log10((2**(i*0.5))/xl)
            30      format(',',f11.6,$)                    
        end do
    end if
    
    write(81,*) !new line
    write(81,40) frame
    40  format(i11,$)    
    do i=1, in_fradim_satom_ball_maxres*2
        write(81,50) log10(real(counter_all(i)))
        50    format(',',f11.6,$)    
    end do

    write(81,*) !new line
    write(81,60) frame
    60  format(i11,$)    
    do i=1, in_fradim_satom_ball_maxres*2
        write(81,70) log10(real(counter_surf_only(i)))
        70    format(',',f11.6,$)    
    end do

    !Deallocate 
    DEALLOCATE(cell)
    DEALLOCATE(counter_all)
    DEALLOCATE(counter_surf)
    DEALLOCATE(counter_overlap)
    DEALLOCATE(counter_surf_only)
    
    return
end 

!END OF CAL_FRACTAL_SATOM_BALL ROUTINE
!=======================================================================


!=======================================================================
!START OF CAL_FRACTAL_NP_SURF ROUTINE
!Fractal dimension box counting method adapted from:
!Motofumi T. Suzuki, A Three Dimensional Box Counting Method for 
!Measuring Fractal Dimension of 3D Models, The 11th IASTED 
!International Conference on Internet and Multimedia Systems 
!and Applications (IMSA_2007), Hawaii, USA, 08/2007.   
!
!Fractal Dimension = log(Nr) / log (1/r)
!where Nr is the number of occupied grids of length r
!TODO:
!  - Use proper atomic radii

subroutine CAL_FRACTAL_NP_SURF
    
    use KINDS, only: dp
    use VARIABLES, only: frame, headers_done, min_x, min_y, min_z, &
    natoms, xc, yc, zc, xl, yl, zl, flag_surf, in_fradim_np_surf_maxres, &
    ncord, nnlist
    IMPLICIT NONE 
    
    !Local      
    integer :: magn_fac
    integer :: has_surf_neigh
    integer :: i, j, k, l, m, n
    integer :: xt, yt, zt
    integer :: atom_cell_idxx, atom_cell_idxy, atom_cell_idxz
    integer :: scan_cell_idxx, scan_cell_idxy, scan_cell_idxz
    integer :: num_scanx, num_scany, num_scanz
    integer, allocatable :: cell(:,:,:) 
    integer, allocatable :: counter_surf(:), counter_overlap(:), counter_surf_only(:)
    real(dp) :: cell_resx, cell_resy, cell_resz
    real(dp) :: scan_cell_minx, scan_cell_maxx, scan_cell_miny, scan_cell_maxy, scan_cell_minz, scan_cell_maxz
    real(dp) :: scan_cell_nearx, scan_cell_farx, scan_cell_neary, scan_cell_fary, scan_cell_nearz, scan_cell_farz
    real(dp) :: dist_nearest, dist_furthest, atom_rad
    
    ALLOCATE(cell(2**in_fradim_np_surf_maxres,2**in_fradim_np_surf_maxres,2**in_fradim_np_surf_maxres))
    ALLOCATE(counter_surf(in_fradim_np_surf_maxres*2))
    ALLOCATE(counter_overlap(in_fradim_np_surf_maxres*2))
    ALLOCATE(counter_surf_only(in_fradim_np_surf_maxres*2))
    
    cell = 0
    atom_rad = 3.5

    print*,"-Performing box-counting technique on surface atom balls to estimate fractal dimension..."
    do i=1, in_fradim_np_surf_maxres*2

        !zero counters
        counter_surf(i) = 0  
        counter_overlap(i) = 0  
        counter_surf_only(i) = 0  
        
        !determine grid lengths
        magn_fac = 2**(i*0.5)
        cell_resx = xl / magn_fac
        cell_resy = yl / magn_fac
        cell_resz = zl / magn_fac
        print'(a40,i10,f10.3,f10.3,f10.3)','  -Magnification, grid length (x, y, z): ',magn_fac,cell_resx,cell_resy,cell_resz

        !determine number of scans to do around each surface atom  !Yet to be refined!!
        num_scanx = int(2*atom_rad/cell_resx)
        num_scany = int(2*atom_rad/cell_resy)
        num_scanz = int(2*atom_rad/cell_resz)
        !print*,'   -Number of cells to be scanned around atom: ',num_scanx,num_scany,num_scanz

        do j=1, natoms(frame)

            !only scan the boxes around atoms with neighbouring surface atoms
            has_surf_neigh = 0
            do n=1, ncord(j)
                if(flag_surf(nnlist(j,n))==1) has_surf_neigh = 1
            end do
            if(has_surf_neigh==0) cycle

            !identify index of the atom relative to coordinates generated based on current grid length
            atom_cell_idxx = int((xc(j)-min_x)/cell_resx) + 1
            atom_cell_idxy = int((yc(j)-min_y)/cell_resy) + 1
            atom_cell_idxz = int((zc(j)-min_z)/cell_resz) + 1
            !print*,'atom_cell_idxx,atom_cell_idxy,atom_cell_idxz: ',atom_cell_idxx,atom_cell_idxy,atom_cell_idxz

            !scan the cells around the atom
            do k=-num_scanx, num_scanx
                scan_cell_idxx = atom_cell_idxx + k
                if(scan_cell_idxx<1 .or. scan_cell_idxx>magn_fac) cycle
                scan_cell_minx = min_x + scan_cell_idxx*cell_resx - cell_resx
                scan_cell_maxx = min_x + scan_cell_idxx*cell_resx
                if(xc(j)<scan_cell_minx) then
                    scan_cell_nearx = scan_cell_minx
                    scan_cell_farx = scan_cell_maxx
                else if(xc(j)>scan_cell_maxx) then
                    scan_cell_nearx = scan_cell_maxx
                    scan_cell_farx = scan_cell_minx
                else
                    scan_cell_nearx = xc(j)
                    if(scan_cell_maxx-xc(j) < cell_resx/2) then
                        scan_cell_farx = scan_cell_minx
                    else
                        scan_cell_farx = scan_cell_maxx
                    end if
                end if
                !print*,scan_cell_minx,scan_cell_maxx,scan_cell_nearx,xc(j)

                do l=-num_scany, num_scany
                    scan_cell_idxy = atom_cell_idxy + l
                    if(scan_cell_idxy<1 .or. scan_cell_idxy>magn_fac) cycle
                    scan_cell_miny = min_y + scan_cell_idxy*cell_resy - cell_resy
                    scan_cell_maxy = min_y + scan_cell_idxy*cell_resy
                    if(yc(j)<scan_cell_miny) then
                        scan_cell_neary = scan_cell_miny
                        scan_cell_fary = scan_cell_maxy
                    else if(yc(j)>scan_cell_maxy) then
                        scan_cell_neary = scan_cell_maxy
                        scan_cell_fary = scan_cell_miny
                    else
                        scan_cell_neary = yc(j)
                        if(scan_cell_maxy-yc(j) < cell_resy/2) then
                            scan_cell_fary = scan_cell_miny
                        else
                            scan_cell_fary = scan_cell_maxy
                        end if
                    end if

                    do m=-num_scanz, num_scanz
                        scan_cell_idxz = atom_cell_idxz + m
                        if(scan_cell_idxz<1 .or. scan_cell_idxz>magn_fac) cycle
                        scan_cell_minz = min_z + scan_cell_idxz*cell_resz - cell_resz
                        scan_cell_maxz = min_z + scan_cell_idxz*cell_resz
                        if(zc(j)<scan_cell_minz) then
                            scan_cell_nearz = scan_cell_minz
                            scan_cell_farz = scan_cell_maxz
                        else if(zc(j)>scan_cell_maxz) then
                            scan_cell_nearz = scan_cell_maxz
                            scan_cell_farz = scan_cell_minz
                        else
                            scan_cell_nearz = zc(j)
                            if(scan_cell_maxz-zc(j) < cell_resz/2) then
                                scan_cell_farz = scan_cell_minz
                            else
                                scan_cell_farz = scan_cell_maxz
                            end if
                        end if

                        !check whether the cell covers the surface
                        dist_nearest = sqrt((xc(j)-scan_cell_nearx)**2 + (yc(j)-scan_cell_neary)**2 + (zc(j)-scan_cell_nearz)**2)
                        dist_furthest = sqrt((xc(j)-scan_cell_farx)**2 + (yc(j)-scan_cell_fary)**2 + (zc(j)-scan_cell_farz)**2)
                        if(flag_surf(j)==0) then  !for bulk atom
                            if(dist_nearest<atom_rad) then
                                if(cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz)==0) then  !if unscanned, mark as bulk
                                    cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz) = 1
                                else if(cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz)==1) then  !skip if already marked as bulk
                                    cycle
                                else if(cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz)==2) then  !if marked as surf, mark as overlap
                                    cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz) = 3
                                    counter_overlap(i) = counter_overlap(i) + 1
                                else if(cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz)==3) then  !skip if already marked as overlap
                                    cycle
                                end if
                            end if
                        else if(flag_surf(j)==1) then  !for surface atom
                            if(dist_nearest<atom_rad .and. dist_furthest>atom_rad) then
                                if(cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz)==0) then  !if unscanned, mark as surf
                                    cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz) = 2
                                    counter_surf(i) = counter_surf(i) + 1
                                else if(cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz)==1) then  !if marked as bulk, mark as overlap
                                    cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz) = 3
                                    counter_overlap(i) = counter_overlap(i) + 1
                                    counter_surf(i) = counter_surf(i) + 1
                                else if(cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz)==2) then  !skip if already marked as surf
                                    cycle
                                else if(cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz)==3) then  !skip if already marked as overlap
                                    cycle
                                end if
                            else if(dist_nearest<atom_rad) then
                                if(cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz)==0) then  !if unscanned, mark as bulk
                                    cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz) = 1
                                else if(cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz)==1) then  !skip if already marked as bulk
                                    cycle
                                else if(cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz)==2) then  !if marked as surf, mark as overlap
                                    cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz) = 3
                                    counter_overlap(i) = counter_overlap(i) + 1
                                else if(cell(scan_cell_idxx,scan_cell_idxy,scan_cell_idxz)==3) then  !skip if already marked as overlap
                                    cycle
                                end if
                            end if
                        end if   
                    end do
                end do
            end do
        end do
        
        !Zero cell array        
        cell = 0
        print*,' -Counts (surf, overlap):            ',counter_surf(i),counter_overlap(i)
        
    end do
    counter_surf_only = counter_surf - counter_overlap
    
    !Write out data
    if(headers_done==0) then
        write(82,10)
        10    format('FRACTAL DIMENSION (2nd line: LOG[1/r]  Below 2nd line: LOG[no of grids occupied])') 
        write(82,20)
        20    format('      Frame',$)        
        do i=1, in_fradim_np_surf_maxres*2
            write(82,30) log10((2**(i*0.5))/xl)
            30      format(',',f11.6,$)                    
        end do
    end if

    write(82,*) !new line
    write(82,60) frame
    60  format(i11,$)    
    do i=1, in_fradim_np_surf_maxres*2
        write(82,70) log10(real(counter_surf_only(i)))
        70    format(',',f11.6,$)    
    end do

    !Deallocate 
    DEALLOCATE(cell)
    DEALLOCATE(counter_surf)
    DEALLOCATE(counter_overlap)
    DEALLOCATE(counter_surf_only)
    
    return
end 

!END OF CAL_FRACTAL_NP_SURF ROUTINE
!=======================================================================


!=======================================================================
!START OF CAL_LINDEMANN
!Lindemann index calculation

subroutine CAL_LINDEMANN
    
    use KINDS, only: dp
    use VARIABLES, only: c_laba, framecount,                           &
    in_lindem_frames, in_write_screen, lca,                            &
    natoms, natoms_max, typecount, xca, yca, zca, xl, yl, zl,          &
    xl2inv, yl2inv, zl2inv
    IMPLICIT NONE 
    
    
    integer :: i, j, k
    integer :: tloop, i1, i2
    integer :: counter
    
    integer :: fr_start, fr_end 
    integer :: totalframes
    
    real(dp) :: totalsteps
    real(dp) :: disx, disy, disz, disr, disrsq
    real(dp) :: disrsq_sum, disr_sum, disr_sumsq
    real(dp) :: term1, term2
    real(dp) :: linde_i, linde_t  
    
    integer, allocatable :: labatom(:,:)      
    
    real(dp), allocatable :: termscale_i(:)
    real(dp), allocatable :: termscale_t(:)
    real(dp), allocatable :: lin_atom(:,:)                              !Lindemann index in frame i of particle j
    real(dp), allocatable :: lin_type(:,:)                              !Average Lindemann index in frame i of particle type j              
    real(dp), allocatable :: lin_syst(:)                                !Average Lindemann index of frame i
    real(dp) progress    
    
    
    if(in_write_screen==1) then
        print*,'-CAL: Lindemann Index'   
    end if  
    

    
    !number of frames to average time over (prior to and after)
    !      eg. if lindemann_frames = 3 then F(-3),F(-2),F(-1),F(current),F(+1),F(+2),F(+3)
    !      where F(current) is the frame for which the index is being calculated for
    
    
    !frames range at which time the index is calculated     
    fr_start = in_lindem_frames + 1                
    fr_end   = framecount - in_lindem_frames
    totalframes = (fr_end - fr_start) + 1                               !total frames in analysis
    totalsteps = in_lindem_frames + in_lindem_frames + 1                !average is either side of tloop by lindemann_frames amount     
    
    !allocate arrays
    ALLOCATE(lin_atom(framecount,natoms_max)) 
    ALLOCATE(lin_type(framecount,typecount))
    ALLOCATE(lin_syst(framecount)) 
    ALLOCATE(labatom(framecount,natoms_max))
    ALLOCATE(termscale_i(framecount))
    ALLOCATE(termscale_t(framecount))
    
    !pre-calculate scaling constants
    do i=1, framecount
        termscale_i(i)  = 1.0 / (natoms(i) - 1.0)                       !1/(N-1)
        termscale_t(i)  = 1.0 / natoms(i)                               !1/N    
    end do           
    
    
    !Zero arrays 
    lin_atom  = 0.0
    lin_syst  = 0.0 
    
    !ERROR - can't calculate index for single configuration
    if(framecount==1) then
        print*,' '
        print*,'ERROR: Cant average Lindemann index over a single frame'
        STOP
    end if 
    
    !ERROR - must have time average
    if(in_lindem_frames==0) then
        print*,' '
        print*,'ERROR: Cant average Lindemann index over a single frame'
        STOP
    end if       
    
    
    !ERROR - ensure enough frames for averaging
    if(fr_start>fr_end) then
        print*,'ERROR: Not enough frames for averaging selection of lindemann_frames'
        STOP
    end if
    
    
    !ERROR - need constant number of atoms in all frames
    do i=1, framecount
        do j=1, framecount
            if((natoms(i)-natoms(j))/=0) then
                print*,'ERROR: Number of particles in all frames must be constant'
                STOP 
            end if
        end do
    end do    
    
    
    
    
    !Main loop of program
    do tloop = fr_start, fr_end                                         !index calculation at time centred upon tloop 
        
        !       -write out progress to screen
        i1 = tloop-fr_start+1
        i2 = fr_end-fr_start+1
        
        if(i2<20) then    !2 X 10
            if(mod(i1,2)==0) then
                progress = real(i1)/real(i2)*100.0
                if(in_write_screen==1) then
                    print*,' --Progress:',progress,'%'
                end if 
            end if
        else
            if(mod(i1,i2/10)==0) then
                progress = real(i1)/real(i2)*100.0
                if(in_write_screen==1) then
                    print*,' --Progress:',progress,'%'
                end if 
            end if
        end if
        
        
        
        
        linde_t = 0.0                                                   !zero system Lindemann index
        do i=1, natoms(tloop)
            
            
            linde_i = 0.0                                               !zero atom Lindemann index
            do j=1, natoms(tloop)
                if(i/=j) then
                    disrsq_sum = 0.0
                    disr_sum   = 0.0
                    
                    do k= tloop - in_lindem_frames, tloop + in_lindem_frames      !time average
                        
                        disx = xca(k,i) - xca(k,j)
                        disy = yca(k,i) - yca(k,j)
                        disz = zca(k,i) - zca(k,j)          
                        
                        disx = disx - (int(disx*xl2inv)*xl)  
                        disy = disy - (int(disy*yl2inv)*yl)
                        disz = disz - (int(disz*zl2inv)*zl)  
                        
                        disrsq = disx*disx+disy*disy+disz*disz          ! Rij^2
                        disr   = sqrt(disrsq)                           ! Rij                          
                        disrsq_sum = disrsq_sum + disrsq
                        disr_sum   = disr_sum + disr              
                    end do
                    
                    disrsq_sum = disrsq_sum / totalsteps                !<Rij^2>
                    disr_sum   = disr_sum / totalsteps                  !<Rij>
                    disr_sumsq  = disr_sum * disr_sum                   !<Rij>^2
                    
                    !term1 = sqrt(<Rij^2> - <Rij>^2)/<Rij>    
                    term1 = disrsq_sum - disr_sumsq   
                    if(term1<0.0) term1 = 0.0                           !eliminate small negative values 
                    term2 = sqrt(term1)/disr_sum
                    
                    linde_i = linde_i + term2
                end if
                
            end do   
            linde_i = linde_i * termscale_i(tloop)                      !Lindemann index of atom i 
            lin_atom(tloop,i) = linde_i                                 !avg Lindemann index of atom i around frame tloop
            
            
            linde_t = linde_t + linde_i      
        end do
        
        linde_t = linde_t * termscale_t(tloop)                          !Lindemann index of system 
        lin_syst(tloop) = linde_t                                       !Avg Lindemann index of system around frame tloop
    end do
    
    
    
    !Avg Lindemann index of particles types
    lin_type = 0
    do i=1, framecount
        do j=1, typecount
            counter = 0
            do k=1, natoms(i)
                
                if(c_laba(i,k)==j) then
                    lin_type(i,j) = lin_type(i,j) + lin_atom(i,k)
                    counter = counter + 1  
                end if
                
            end do
            if(counter>0) then
                lin_type(i,j) = lin_type(i,j) / counter                 !Average 
            end if
            
            
        end do
    end do
    
    

    !Write out system index
    write(12,10)
    10  format('LINDEMANN INDEX AVERAGE FOR SYSTEM')
    write(12,*)
    write(12,20)
    20  format('Frame      System       Type1        Type2        Type3')   
    do i=1, framecount
        write(12,30) i, lin_syst(i)
        30    format(i5,f12.6,$)
        do j=1, typecount
            write(12,40) lin_type(i,j)
            40    format(f12.6,$)
        end do
        write(12,*) 
    end do
    
    
    
    
    !Write out local LI index
    do i=fr_start, fr_end
        write(13,'(i6,/)') natoms(i) 
        
        do j=1, natoms(i)         
            write(13,'(a,2x,4(F10.5,2X))') lca(i,j),xca(i,j),yca(i,j),zca(i,j),lin_atom(i,j)                             
        end do 
    end do   
    

    
    !Deallocate arrays
    DEALLOCATE(lin_atom) 
    DEALLOCATE(lin_type)
    DEALLOCATE(lin_syst) 
    DEALLOCATE(labatom)
    DEALLOCATE(termscale_i)
    DEALLOCATE(termscale_t)      
    
  
    
    return
end 



!END OF CAL_LINDEMANN ROUTINE
!=======================================================================



!=======================================================================
!START OF SPC_SLICEXYZ ROUTINE

!Slice out 5 rectangular volumes from a cell in the z direction
!remaining periodic (if PBC used) in x and y directions.  Only
!done for single configuration (last one)

subroutine SPC_SLICEXYZ
    
    use KINDS, only: dp
    use VARIABLES, only: frame, natoms, xc, yc, zc, xyz_prec_str
    IMPLICIT NONE
        
    
    !Local
    integer :: i
    integer :: counter
    real(dp) :: slice_width
    real(dp) :: x_temp, y_temp, z_temp
    character(len=3) labchar
    
    !slice_width is from origin to slice_width along z direction
    slice_width = 200      
    
    !label for output xyz
    labchar = 'AA'      
    
    
    !SLICE 1      
    !count total atoms needed
    counter = 0
    do i=1, natoms(frame)
        x_temp = xc(i)
        y_temp = yc(i)
        z_temp = zc(i)
        if(z_temp<slice_width) then
            counter = counter + 1
        end if        
    end do
    
    open(1,status="unknown",file='PRJ_Slicexyz1.xyz')  
    
    write(1,'(i10,/)') counter
    do i=1, natoms(frame)
        x_temp = xc(i)
        y_temp = yc(i)
        z_temp = zc(i)
        
        if(z_temp<slice_width) then
            write(1,'(a,2x,3(F10.'//xyz_prec_str//',2X))') labchar,x_temp,y_temp,z_temp 
        end if
    end do
    close(1)
    
    
    
    !SLICE 2      
    !count total atoms needed
    counter = 0
    do i=1, natoms(frame)
        x_temp = xc(i)
        y_temp = yc(i)
        z_temp = zc(i)
        if(z_temp>slice_width.and.z_temp<2*slice_width) then
            counter = counter + 1
        end if        
    end do
    
    open(1,status="unknown",file='PRJ_Slicexyz2.xyz')  
    
    write(1,'(i10,/)') counter
    do i=1, natoms(frame)
        x_temp = xc(i)
        y_temp = yc(i)
        z_temp = zc(i)
        
        if(z_temp>slice_width.and.z_temp<2*slice_width) then
            write(1,'(a,2x,3(F10.'//xyz_prec_str//',2X))') labchar,x_temp,y_temp,z_temp 
        end if
    end do
    close(1)    
    
    
    
    !SLICE 3      
    !count total atoms needed
    counter = 0
    do i=1, natoms(frame)
        x_temp = xc(i)
        y_temp = yc(i)
        z_temp = zc(i)
        if(z_temp>2*slice_width.and.z_temp<3*slice_width) then
            counter = counter + 1
        end if        
    end do
    
    open(1,status="unknown",file='PRJ_Slicexyz3.xyz')  
    
    write(1,'(i10,/)') counter
    do i=1, natoms(frame)
        x_temp = xc(i)
        y_temp = yc(i)
        z_temp = zc(i)
        
        if(z_temp>2*slice_width.and.z_temp<3*slice_width) then
            write(1,'(a,2x,3(F10.'//xyz_prec_str//',2X))') labchar,x_temp,y_temp,z_temp 
        end if
    end do
    close(1)    
    
    
    
    !SLICE 4      
    !count total atoms needed
    counter = 0
    do i=1, natoms(frame)
        x_temp = xc(i)
        y_temp = yc(i)
        z_temp = zc(i)
        if(z_temp>3*slice_width.and.z_temp<4*slice_width) then
            counter = counter + 1
        end if        
    end do
    
    open(1,status="unknown",file='PRJ_Slicexyz4.xyz')  
    
    write(1,'(i10,/)') counter
    do i=1, natoms(frame)
        x_temp = xc(i)
        y_temp = yc(i)
        z_temp = zc(i)
        
        if(z_temp>3*slice_width.and.z_temp<4*slice_width) then
            write(1,'(a,2x,3(F10.'//xyz_prec_str//',2X))') labchar,x_temp,y_temp,z_temp 
        end if
    end do
    close(1)   
    
    
    
    !SLICE 5      
    !count total atoms needed
    counter = 0
    do i=1, natoms(frame)
        x_temp = xc(i)
        y_temp = yc(i)
        z_temp = zc(i)
        if(z_temp>4*slice_width.and.z_temp<5*slice_width) then
            counter = counter + 1
        end if        
    end do
    
    open(1,status="unknown",file='PRJ_Slicexyz5.xyz')  
    
    write(1,'(i10,/)') counter
    do i=1, natoms(frame)
        x_temp = xc(i)
        y_temp = yc(i)
        z_temp = zc(i)
        
        if(z_temp>4*slice_width.and.z_temp<5*slice_width) then
            write(1,'(a,2x,3(F10.'//xyz_prec_str//',2X))') labchar,x_temp,y_temp,z_temp 
        end if
    end do
    close(1)       
    
    
    return
end 


!END OF SPC_SLICEXYZ ROUTINE
!=======================================================================



!=======================================================================
!START OF OUT_PROGRESS ROUTINE
!write out progress to screen

subroutine OUT_PROGRESS
    
    use KINDS, only: dp
    use VARIABLES, only: frame, in_frames_end, in_write_screen     
    IMPLICIT NONE
    
    
    !Local    
    integer :: i1, i2
    real(dp) progress
    
    
    
    
    i1 = frame
    i2 = in_frames_end
    
    if(i2<20) then    !2 X 10
        if(mod(i1,2)==0) then
            progress = real(i1)/real(i2)*100.0
            if(in_write_screen==1) then
                print*,'--Frames Analyzed:',progress,'%'
            end if 
        end if
    else
        if(mod(i1,i2/10)==0) then
            progress = real(i1)/real(i2)*100.0
            if(in_write_screen==1) then
                print*,'--Frames Analyzed:',progress,'%'
            end if 
        end if
    end if
    
    
    return
end 


!END OF OUT_PROGRESS ROUTINE
!=======================================================================



!=======================================================================
!START OF TABLE_RCOVALENT ROUTINE

subroutine TABLE_RCOVALENT
    
    use VARIABLES    
    IMPLICIT NONE 
    
    real(dp) :: c_rcov(300,10)                                          !covalent radii of element i, type j (variations like sp,sp2 etc)
    real(dp) :: c_dcov(300,10)                                          !covalent radii standard deviation
    character(len=2) :: c_elem(300)  
    
    
    
    !elemental types as function of atomic number
    c_elem(1)  = 'H '
    c_elem(2)  = 'He'
    c_elem(3)  = 'Li'
    c_elem(4)  = 'Be'
    c_elem(5)  = 'B '
    c_elem(6)  = 'C '   
    c_elem(7)  = 'N '
    c_elem(8)  = 'O '
    c_elem(9)  = 'F '
    c_elem(10) = 'Ne'
    c_elem(11) = 'Na'
    c_elem(12) = 'Mg'
    c_elem(13) = 'Al'
    c_elem(14) = 'Si'
    c_elem(15) = 'P '   
    c_elem(16) = 'S '
    c_elem(17) = 'Cl'
    c_elem(18) = 'Ar'
    c_elem(19) = 'K '
    c_elem(20) = 'Ca'
    c_elem(21) = 'Sc'
    c_elem(22) = 'Ti'
    c_elem(23) = 'V '
    c_elem(24) = 'Cr'
    c_elem(25) = 'Mn'       
    c_elem(26) = 'Fe' 
    c_elem(27) = 'Co'
    c_elem(28) = 'Ni'
    c_elem(29) = 'Cu'
    c_elem(30) = 'Zn'
    c_elem(31) = 'Ga'
    c_elem(32) = 'Ge'    
    c_elem(33) = 'As'
    c_elem(34) = 'Se'
    c_elem(35) = 'Br'
    c_elem(36) = 'Kr'
    c_elem(37) = 'Rb'
    c_elem(38) = 'Sr'
    c_elem(39) = 'Y '
    c_elem(40) = 'Zr'
    c_elem(41) = 'Nb'
    c_elem(42) = 'Mo'
    c_elem(43) = 'Tc'
    c_elem(44) = 'Ru'
    c_elem(45) = 'Rh'
    c_elem(46) = 'Pd'
    c_elem(47) = 'Ag'
    c_elem(48) = 'Cd'
    c_elem(49) = 'In'
    c_elem(50) = 'Sn'
    c_elem(51) = 'Sb'
    c_elem(52) = 'Te'
    c_elem(53) = 'I '
    c_elem(54) = 'Xe'
    c_elem(55) = 'Cs'
    c_elem(56) = 'Ba'
    c_elem(57) = 'La'
    c_elem(58) = 'Ce'
    c_elem(59) = 'Pr'
    c_elem(60) = 'Nd'
    c_elem(61) = 'Pm'
    c_elem(62) = 'Sm'
    c_elem(63) = 'Eu'
    c_elem(64) = 'Gd'
    c_elem(65) = 'Tb'
    c_elem(66) = 'Dy'   
    c_elem(67) = 'Ho'
    c_elem(68) = 'Er'
    c_elem(69) = 'Tm'
    c_elem(70) = 'Yb'
    c_elem(71) = 'Lu'
    c_elem(72) = 'Hf'
    c_elem(73) = 'Ta'
    c_elem(74) = 'W '
    c_elem(75) = 'Re'
    c_elem(76) = 'Os'
    c_elem(77) = 'Ir'
    c_elem(78) = 'Pt'
    c_elem(79) = 'Au'
    c_elem(80) = 'Hg'
    c_elem(81) = 'Tl'
    c_elem(82) = 'Pb'
    c_elem(83) = 'Bi' 
    c_elem(84) = 'Po'
    c_elem(85) = 'At'
    c_elem(86) = 'Rn'
    c_elem(87) = 'Fr'
    c_elem(88) = 'Ra'
    c_elem(89) = 'Ac'
    c_elem(90) = 'Th'
    c_elem(91) = 'Pa'
    c_elem(92) = 'U '
    c_elem(93) = 'Np'
    c_elem(94) = 'Pu'
    c_elem(95) = 'Am'
    c_elem(96) = 'Cm'     
    
    
    
    
    !Constants for the covalent radius of elements.  Taken from 
    !B. Cordero et al, Dalton Transactions, 2823-2838 (2008). 
    
    c_rcov(1,1)  = 0.31   !H
    c_rcov(2,1)  = 0.28   !He
    c_rcov(3,1)  = 1.28   !Li
    c_rcov(4,1)  = 0.96   !Be
    c_rcov(5,1)  = 0.84   !B
    c_rcov(6,1)  = 0.76   !C sp3
    c_rcov(6,2)  = 0.73   !C sp2
    c_rcov(6,3)  = 0.69   !C sp1      
    c_rcov(7,1)  = 0.71   !N
    c_rcov(8,1)  = 0.66   !O
    c_rcov(9,1)  = 0.57   !F
    c_rcov(10,1) = 0.58   !Ne
    c_rcov(11,1) = 1.66   !Na
    c_rcov(12,1) = 1.41   !Mg
    c_rcov(13,1) = 1.21   !Al
    c_rcov(14,1) = 1.11   !Si
    c_rcov(15,1) = 1.07   !P   
    c_rcov(16,1) = 1.05   !S
    c_rcov(17,1) = 1.02   !Cl
    c_rcov(18,1) = 1.06   !Ar
    c_rcov(19,1) = 2.03   !K
    c_rcov(20,1) = 1.76   !Ca
    c_rcov(21,1) = 1.70   !Sc
    c_rcov(22,1) = 1.60   !Ti
    c_rcov(23,1) = 1.53   !V
    c_rcov(24,1) = 1.39   !Cr
    c_rcov(25,1) = 1.61   !Mn h.s (high spin)
    c_rcov(25,2) = 1.39   !Mn l.s (low spin)          
    c_rcov(26,1) = 1.52   !Fe h.s
    c_rcov(26,2) = 1.32   !Fe l.s
    c_rcov(27,1) = 1.50   !Co h.s
    c_rcov(27,2) = 1.26   !Co l.s
    c_rcov(28,1) = 1.24   !Ni
    c_rcov(29,1) = 1.32   !Cu
    c_rcov(30,1) = 1.22   !Zn
    c_rcov(31,1) = 1.22   !Ga
    c_rcov(32,1) = 1.20   !Ge    
    c_rcov(33,1) = 1.19   !As
    c_rcov(34,1) = 1.20   !Se
    c_rcov(35,1) = 1.20   !Br
    c_rcov(36,1) = 1.16   !Kr
    c_rcov(37,1) = 2.20   !Rb
    c_rcov(38,1) = 1.95   !Sr
    c_rcov(39,1) = 1.90   !Y
    c_rcov(40,1) = 1.75   !Zr
    c_rcov(41,1) = 1.64   !Nb
    c_rcov(42,1) = 1.54   !Mo
    c_rcov(43,1) = 1.47   !Tc
    c_rcov(44,1) = 1.46   !Ru
    c_rcov(45,1) = 1.42   !Rh
    c_rcov(46,1) = 1.39   !Pd
    c_rcov(47,1) = 1.45   !Ag
    c_rcov(48,1) = 1.44   !Cd
    c_rcov(49,1) = 1.42   !In  
    c_rcov(50,1) = 1.39   !Sn
    c_rcov(51,1) = 1.39   !Sb
    c_rcov(52,1) = 1.38   !Te
    c_rcov(53,1) = 1.39   !I
    c_rcov(54,1) = 1.40   !Xe
    c_rcov(55,1) = 2.44   !Cs
    c_rcov(56,1) = 2.15   !Ba
    c_rcov(57,1) = 2.07   !La
    c_rcov(58,1) = 2.04   !Ce
    c_rcov(59,1) = 2.03   !Pr
    c_rcov(60,1) = 2.01   !Nd
    c_rcov(61,1) = 1.99   !Pm
    c_rcov(62,1) = 1.98   !Sm
    c_rcov(63,1) = 1.98   !Eu
    c_rcov(64,1) = 1.96   !Gd
    c_rcov(65,1) = 1.94   !Tb
    c_rcov(66,1) = 1.92   !Dy    
    c_rcov(67,1) = 1.92   !Ho
    c_rcov(68,1) = 1.89   !Er
    c_rcov(69,1) = 1.90   !Tm
    c_rcov(70,1) = 1.87   !Yb
    c_rcov(71,1) = 1.87   !Lu
    c_rcov(72,1) = 1.75   !Hf
    c_rcov(73,1) = 1.70   !Ta
    c_rcov(74,1) = 1.62   !W
    c_rcov(75,1) = 1.51   !Re
    c_rcov(76,1) = 1.44   !Os
    c_rcov(77,1) = 1.41   !Ir
    c_rcov(78,1) = 1.36   !Pt
    c_rcov(79,1) = 1.36   !Au
    c_rcov(80,1) = 1.32   !Hg
    c_rcov(81,1) = 1.45   !Tl
    c_rcov(82,1) = 1.46   !Pb
    c_rcov(83,1) = 1.48   !Bi  
    c_rcov(84,1) = 1.40   !Po
    c_rcov(85,1) = 1.50   !At
    c_rcov(86,1) = 1.50   !Rn
    c_rcov(87,1) = 2.60   !Fr
    c_rcov(88,1) = 2.21   !Ra
    c_rcov(89,1) = 2.15   !Ac
    c_rcov(90,1) = 2.06   !Th
    c_rcov(91,1) = 2.00   !Pa
    c_rcov(92,1) = 1.96   !U
    c_rcov(93,1) = 1.90   !Np
    c_rcov(94,1) = 1.87   !Pu
    c_rcov(95,1) = 1.80   !Am
    c_rcov(96,1) = 1.69   !Cm
    
    
    !Constants for the standard deviation of the radius histogram of 
    !elements.  Taken from B. Cordero et al, Dalton Transactions, 2823-2838 (2008).      
    
    c_dcov(1,1)  = 0.05   !H
    c_dcov(2,1)  = 0.05   !He   !value not in ref
    c_dcov(3,1)  = 0.07   !Li
    c_dcov(4,1)  = 0.03   !Be
    c_dcov(5,1)  = 0.03   !B
    c_dcov(6,1)  = 0.01   !C sp3
    c_dcov(6,2)  = 0.02   !C sp2
    c_dcov(6,3)  = 0.01   !C sp1      
    c_dcov(7,1)  = 0.01   !N
    c_dcov(8,1)  = 0.02   !O
    c_dcov(9,1)  = 0.03   !F
    c_dcov(10,1) = 0.05   !Ne   !value=not in ref
    c_dcov(11,1) = 0.09   !Na
    c_dcov(12,1) = 0.07   !Mg
    c_dcov(13,1) = 0.04   !Al
    c_dcov(14,1) = 0.02   !Si
    c_dcov(15,1) = 0.03   !P   
    c_dcov(16,1) = 0.03   !S
    c_dcov(17,1) = 0.04   !Cl
    c_dcov(18,1) = 0.10   !Ar
    c_dcov(19,1) = 0.12   !K
    c_dcov(20,1) = 0.10   !Ca
    c_dcov(21,1) = 0.07   !Sc
    c_dcov(22,1) = 0.08   !Ti
    c_dcov(23,1) = 0.08   !V
    c_dcov(24,1) = 0.05   !Cr
    c_dcov(25,1) = 0.08   !Mn h.s (high spin)
    c_dcov(25,2) = 0.05   !Mn l.s (low spin)          
    c_dcov(26,1) = 0.06   !Fe h.s
    c_dcov(26,2) = 0.03   !Fe l.s
    c_dcov(27,1) = 0.07   !Co h.s
    c_dcov(27,2) = 0.03   !Co l.s
    c_dcov(28,1) = 0.04   !Ni
    c_dcov(29,1) = 0.04   !Cu
    c_dcov(30,1) = 0.04   !Zn
    c_dcov(31,1) = 0.03   !Ga
    c_dcov(32,1) = 0.04   !Ge    
    c_dcov(33,1) = 0.04   !As
    c_dcov(34,1) = 0.04   !Se
    c_dcov(35,1) = 0.03   !Br
    c_dcov(36,1) = 0.04   !Kr
    c_dcov(37,1) = 0.09   !Rb
    c_dcov(38,1) = 0.10   !Sr
    c_dcov(39,1) = 0.07   !Y
    c_dcov(40,1) = 0.07   !Zr
    c_dcov(41,1) = 0.06   !Nb
    c_dcov(42,1) = 0.05   !Mo
    c_dcov(43,1) = 0.07   !Tc
    c_dcov(44,1) = 0.07   !Ru
    c_dcov(45,1) = 0.07   !Rh
    c_dcov(46,1) = 0.06   !Pd
    c_dcov(47,1) = 0.05   !Ag
    c_dcov(48,1) = 0.09   !Cd
    c_dcov(49,1) = 0.05   !In  
    c_dcov(50,1) = 0.04   !Sn
    c_dcov(51,1) = 0.05   !Sb
    c_dcov(52,1) = 0.04   !Te
    c_dcov(53,1) = 0.03   !I
    c_dcov(54,1) = 0.09   !Xe
    c_dcov(55,1) = 0.11   !Cs
    c_dcov(56,1) = 0.11   !Ba
    c_dcov(57,1) = 0.08   !La
    c_dcov(58,1) = 0.09   !Ce
    c_dcov(59,1) = 0.07   !Pr
    c_dcov(60,1) = 0.06   !Nd
    c_dcov(61,1) = 0.10   !Pm   !value not in ref
    c_dcov(62,1) = 0.08   !Sm
    c_dcov(63,1) = 0.06   !Eu
    c_dcov(64,1) = 0.06   !Gd
    c_dcov(65,1) = 0.05   !Tb
    c_dcov(66,1) = 0.07   !Dy    
    c_dcov(67,1) = 0.07   !Ho
    c_dcov(68,1) = 0.06   !Er
    c_dcov(69,1) = 0.10   !Tm
    c_dcov(70,1) = 0.08   !Yb
    c_dcov(71,1) = 0.08   !Lu
    c_dcov(72,1) = 0.10   !Hf
    c_dcov(73,1) = 0.08   !Ta
    c_dcov(74,1) = 0.07   !W
    c_dcov(75,1) = 0.07   !Re
    c_dcov(76,1) = 0.04   !Os
    c_dcov(77,1) = 0.06   !Ir
    c_dcov(78,1) = 0.05   !Pt
    c_dcov(79,1) = 0.06   !Au
    c_dcov(80,1) = 0.05   !Hg
    c_dcov(81,1) = 0.07   !Tl
    c_dcov(82,1) = 0.05   !Pb
    c_dcov(83,1) = 0.04   !Bi  
    c_dcov(84,1) = 0.04   !Po
    c_dcov(85,1) = 0.10   !At   !value not in ref
    c_dcov(86,1) = 0.10   !Rn   !value not in ref  
    c_dcov(87,1) = 0.10   !Fr   !value not in ref
    c_dcov(88,1) = 0.02   !Ra
    c_dcov(89,1) = 0.10   !Ac   !value not in ref
    c_dcov(90,1) = 0.06   !Th
    c_dcov(91,1) = 0.10   !Pa   !value not in ref
    c_dcov(92,1) = 0.07   !U
    c_dcov(93,1) = 0.01   !Np
    c_dcov(94,1) = 0.01   !Pu
    c_dcov(95,1) = 0.06   !Am
    c_dcov(96,1) = 0.03   !Cm
    
 
    return
end 



!END OF TABLE_RCOVALENT ROUTINE
!=======================================================================
