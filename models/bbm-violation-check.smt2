(declare-const BBM_Wl0 Event)
(declare-const BBM_Wl1 Event)
(declare-const BBM_Wl2 Event)
(declare-const BBM_Wl3 Event)

(declare-const BBM_Wl0_pa (_ BitVec 64))
(declare-const BBM_Wl1_pa (_ BitVec 64))
(declare-const BBM_Wl2_pa (_ BitVec 64))
(declare-const BBM_Wl3_pa (_ BitVec 64))

(assert (not (= BBM_Wl0_pa BBM_Wl1_pa)))
(assert (not (= BBM_Wl1_pa BBM_Wl2_pa)))
(assert (not (= BBM_Wl2_pa BBM_Wl3_pa)))

(declare-const BBM_Wl0_data (_ BitVec 64))
(declare-const BBM_Wl1_data (_ BitVec 64))
(declare-const BBM_Wl2_data (_ BitVec 64))
(declare-const BBM_Wl3_data (_ BitVec 64))

(declare-const BBM_ia (_ BitVec 36))

(define-fun ia_offset3 ((ia (_ BitVec 36))) (_ BitVec 12)
  (concat ((_ extract 8 0) ia) #b000))

(define-fun ia_offset2 ((ia (_ BitVec 36))) (_ BitVec 12)
  (concat ((_ extract 17 9) ia) #b000))

(define-fun ia_offset1 ((ia (_ BitVec 36))) (_ BitVec 12)
  (concat ((_ extract 26 18) ia) #b000))

(define-fun ia_offset0 ((ia (_ BitVec 36))) (_ BitVec 12)
  (concat ((_ extract 35 27) ia) #b000))

(define-fun page_offset ((pa (_ BitVec 64))) (_ BitVec 12)
  ((_ extract 11 0) pa))

(define-fun table_address ((desc (_ BitVec 64))) (_ BitVec 64)
  (concat #x0000 ((_ extract 47 12) desc) #x000))

(assert (= (page_offset BBM_Wl0_pa) (ia_offset0 BBM_ia)))
(assert (= (page_offset BBM_Wl1_pa) (ia_offset1 BBM_ia)))
(assert (= (page_offset BBM_Wl2_pa) (ia_offset2 BBM_ia)))
(assert (= (page_offset BBM_Wl3_pa) (ia_offset2 BBM_ia)))

(assert (tt_write BBM_Wl0 BBM_Wl0_pa BBM_Wl0_data))

(define-fun valid_desc ((desc (_ BitVec 64))) Bool
   (= (bvand desc #x0000000000000001) #x0000000000000001))

(define-fun valid_table_desc ((desc (_ BitVec 64))) Bool
   (= (bvand desc #x0000000000000011) #x0000000000000011))

; For each level, if its valid its parent must be a valid table entry
(assert
  (and
    (implies (valid_desc BBM_Wl3_data) (valid_table_desc BBM_Wl2_data))
    (implies (valid_desc BBM_Wl2_data) (valid_table_desc BBM_Wl1_data))
    (implies (valid_desc BBM_Wl1_data) (valid_table_desc BBM_Wl0_data))))

; If an entry is pointed to by its parent, then it must be actually
; represent a valid page table write at the correct location. The
; alternative is if the parent is invalid, in which case anything
; goes
(assert
  (implies (valid_table_desc BBM_Wl0_data)
    (and (tt_write BBM_Wl1 BBM_Wl1_pa BBM_Wl1_data)
         (= (table_address BBM_Wl0_data) (table_address BBM_Wl1_pa)))))

(assert
  (implies (valid_table_desc BBM_Wl1_data)
    (and (tt_write BBM_Wl2 BBM_Wl2_pa BBM_Wl2_data)
         (= (table_address BBM_Wl1_data) (table_address BBM_Wl2_pa)))))

(assert
  (implies (valid_table_desc BBM_Wl2_data)
    (and (tt_write BBM_Wl3 BBM_Wl3_pa BBM_Wl3_data)
         (= (table_address BBM_Wl2_data) (table_address BBM_Wl3_pa)))))

(declare-const BBM_W1 Event)
(declare-const BBM_W1_pa (_ BitVec 64))
(declare-const BBM_W1_data (_ BitVec 64))

(declare-const BBM_W2 Event)

; BBM_W1 and BBM_W2 conflict
(assert (and (tt_write BBM_W1 BBM_W1_pa BBM_W1_data) (valid_desc BBM_W1_data)))
(assert (W_valid BBM_W2))
(assert (not (= ((_ extract 47 12) BBM_W1_data) ((_ extract 47 12) (val_of_64 BBM_W2)))))
(assert (= BBM_W1_pa (addr_of BBM_W2)))

(assert (or
  (and (= BBM_W1 BBM_Wl3) (= BBM_W1_pa BBM_Wl3_pa) (= BBM_W1_data BBM_Wl3_data))
  (and (= BBM_W1 BBM_Wl2) (= BBM_W1_pa BBM_Wl2_pa) (= BBM_W1_data BBM_Wl2_data))
  (and (= BBM_W1 BBM_Wl1) (= BBM_W1_pa BBM_Wl1_pa) (= BBM_W1_data BBM_Wl1_data))
  (and (= BBM_W1 BBM_Wl0) (= BBM_W1_pa BBM_Wl0_pa) (= BBM_W1_data BBM_Wl0_data))))

(assert (co BBM_W1 BBM_W2))

(define-fun BBM_sequence1 ((S_Wp Event) (S_tlbi Event)) Bool
  (and
    (wco BBM_W1 S_Wp)
    (W_invalid S_Wp)
    (implies (= BBM_W1 BBM_Wl3) (or (= S_Wp BBM_Wl3) (= S_Wp BBM_Wl2) (= S_Wp BBM_Wl1) (= S_Wp BBM_Wl0)))
    (implies (= BBM_W1 BBM_Wl2) (or (= S_Wp BBM_Wl2) (= S_Wp BBM_Wl1) (= S_Wp BBM_Wl0)))
    (implies (= BBM_W1 BBM_Wl1) (or (= S_Wp BBM_Wl1) (= S_Wp BBM_Wl0)))
    (implies (= BBM_W1 BBM_Wl0) (= S_Wp BBM_Wl0))
    (wco S_Wp S_tlbi)
    (TLBI-VA S_tlbi)
    (= (tlbi_va (val_of_cache_op S_tlbi)) (concat #x0000 BBM_ia #x000))
    (wco S_tlbi BBM_W2)))

; If there are no valid BBM sequence between BBM_W1 and BBM_W2, we have a BBM violation 
(assert (forall ((BBM_Wp Event) (BBM_tlbi Event))
  (not (BBM_sequence1 BBM_Wp BBM_tlbi))))
