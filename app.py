import os
import pandas as pd
import numpy as np
import streamlit as st
import os
import pandas as pd
import urllib.request
import faiss
import time
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, TFAutoModel
import torch
import re
from datasets import Dataset
import pickle
from collections import defaultdict
from tqdm import tqdm
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from Search import search_documents_with_filter
from streamlit_option_menu import option_menu

# def main_app():
#     st.set_page_config(page_title="Cremong", layout="wide")
#     st.title("Cremong🧸")

#     st.sidebar.title("중복기사 방지 시스템")
#     st.sidebar.subheader("키워드를 입력하면 중복된 기사를 찾아줍니다.")

def main_app():
    st.set_page_config(page_title="Cremond", layout="wide")
    st.title("Cremond🧸")

    # 사이드바 메뉴 설정
    with st.sidebar:
        choice = option_menu("Menu", ["중복기사 방지 시스템", "사용 가이드"],
                             icons=['bi bi-robot', 'question-circle'],
                             menu_icon="app-indicator", default_index=0,
                             styles={
            "container": {"padding": "5!important", "background-color": "#f0f0f0"},
            "icon": {"color": "#333", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#ddd"},
            "nav-link-selected": {"background-color": "#08c7b4"},
        })

    if choice == "중복기사 방지 시스템":
        # 중복기사 방지 시스템 메인 기능
        # st.sidebar.subheader("키워드를 입력하면 중복된 기사를 찾아줍니다.")
        
        # 탭 구성
        tab1, tab2 = st.tabs(["검색 및 결과", "원본 확인"])
        
        with tab1:
            st.subheader("검색 필터")
            
            # 발행연도 필터 (여러 개 선택 가능)
            years = ['2019', '2020', '2021', '2022', '2023', '2024']
            year_filter = st.multiselect("발행연도", options=years, default=years)
            
            # # 발행연월 필터 (여러 개 선택 가능)
            months = [f"{year}-{month:02d}" for year in years for month in range(1, 13)]
            month_filter = st.multiselect("발행연월", options=months)
            
            # 검색창
            search_query = st.text_input("검색어를 입력하세요", placeholder="예: 벤자민 프랭클린")

            # 검색 버튼 클릭 시 검색 수행
            if st.button("검색"):
                if search_query:
                    with st.spinner("검색 중..."):
                        results_df = search_documents_with_filter(search_query, k=30)
                        
                        # 필터링 (발행연도 및 발행연월 기준)
                        if year_filter:
                            results_df = results_df[results_df["발행연도"].isin(year_filter)]
                        
                        # if month_filter:
                        #     results_df = results_df[results_df["발행연월"].isin(month_filter)]
                        
                        if not results_df.empty:
                            st.write("검색 결과:")
                            for _, row in results_df.iterrows():
                                # st.write(f"**발행연도**: {row['발행연도']}, **발행월**: {row['발행연월']}, **페이지**: {row['페이지']}")
                                st.write(f"**발행연도**: {row['발행연도']}, **발행월**: {row['발행월']}, **페이지**: {row['페이지수']}")
                                st.write(f"**내용**: {row['processed']}")
                                st.write(f"**유사도**: {row['유사도']}")
                                st.write("-" * 50)
                        else:
                            st.write("해당 조건에 맞는 결과가 없습니다.")
                else:
                    st.warning("검색어를 입력하세요.")

        with tab2:
            st.subheader("원본 유사 문서 리스트")
            
            if 'results_df' in locals() and not results_df.empty:
                st.dataframe(results_df)
            else:
                st.write("Tab 1에서 검색어를 입력하고 검색 버튼을 눌러주세요.")
    
    elif choice == "사용 가이드":
        st.sidebar.subheader("사용 가이드")
        st.write("1. '중복기사 방지 시스템' 메뉴를 선택합니다.")
        st.write("2. 검색 필터에서 원하는 발행연도와 발행연월을 선택합니다.")
        st.write("3. 검색어를 입력하고 '검색' 버튼을 누르면 관련 기사들을 확인할 수 있습니다.")
    


    # ##############################################
    # # Sidebar 설정
    # # st.sidebar.title("중복기사 방지 시스템")
    # # st.sidebar.subheader("키워드를 입력하면 중복된 기사를 찾아줍니다.")

    # # Sidebar 메뉴 설정
    # st.sidebar.title("중복기사 방지 시스템")
    # # menu_selection = st.sidebar.radio("원하는 기능을 선택하세요:", ["중복기사 방지 시스템", "사용 가이드"])
    # menu_selection = st.sidebar.selectbox("기능 선택:", ["중복기사 방지 시스템", "사용 가이드"])

    # if menu_selection == "중복기사 방지 시스템":
    #     # 중복기사 방지 시스템 메인 기능
    #     # st.sidebar.subheader("키워드를 입력하면 중복된 기사를 찾아줍니다.")
        
    #     # 탭 구성
    #     tab1, tab2 = st.tabs(["검색 및 결과", "원본 확인"])

    #     with tab1:
    #         st.subheader("검색 필터")
            
    #         # 발행연도 필터 (여러 개 선택 가능)
    #         years = ['2019', '2020', '2021', '2022', '2023', '2024']
    #         year_filter = st.multiselect("발행연도", options=years, default=years)
            
    #         # # 발행연월 필터 (여러 개 선택 가능)
    #         # months = [f"{year}-{month:02d}" for year in years for month in range(1, 13)]
    #         # month_filter = st.multiselect("발행연월", options=months)
            
    #         # 검색창
    #         search_query = st.text_input("검색어를 입력하세요", placeholder="예: 벤자민 프랭클린")

    #         # 검색 버튼 클릭 시 검색 수행
    #         if st.button("검색"):
    #             if search_query:
    #                 with st.spinner("검색 중..."):
    #                     results_df = search_documents_with_filter(search_query, k=10)
                        
    #                     # 필터링 (발행연도 및 발행연월 기준)
    #                     if year_filter:
    #                         results_df = results_df[results_df["발행연도"].isin(year_filter)]
                        
    #                     # if month_filter:
    #                     #     results_df = results_df[results_df["발행연월"].isin(month_filter)]
                        
    #                     if not results_df.empty:
    #                         st.write("검색 결과:")
    #                         for _, row in results_df.iterrows():
    #                             # st.write(f"**발행연도**: {row['발행연도']}, **발행월**: {row['발행연월']}, **페이지**: {row['페이지']}")
    #                             st.write(f"**발행연도**: {row['발행연도']}, **발행월**: {row['발행월']}, **페이지**: {row['페이지수']}")
    #                             st.write(f"**내용**: {row['processed']}")
    #                             st.write(f"**유사도**: {row['유사도']}")
    #                             st.write("-" * 50)
    #                     else:
    #                         st.write("해당 조건에 맞는 결과가 없습니다.")
    #             else:
    #                 st.warning("검색어를 입력하세요.")

    #     with tab2:
    #         st.subheader("원본 유사 문서 리스트")
            
    #         if 'results_df' in locals() and not results_df.empty:
    #             st.dataframe(results_df)
    #         else:
    #             st.write("Tab 1에서 검색어를 입력하고 검색 버튼을 눌러주세요.")

    # elif menu_selection == "사용 가이드":
    #     # 사용 가이드 페이지
    #     st.sidebar.subheader("사용 가이드")
    #     st.write("1. '중복기사 방지 시스템' 메뉴를 선택합니다.")
    #     st.write("2. 검색 필터에서 원하는 발행연도와 발행연월을 선택합니다.")
    #     st.write("3. 검색어를 입력하고 '검색' 버튼을 누르면 관련 기사들을 확인할 수 있습니다.")

if __name__ == "__main__":
    main_app()
