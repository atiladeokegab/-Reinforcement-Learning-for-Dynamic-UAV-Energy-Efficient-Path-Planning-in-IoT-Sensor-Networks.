%% Alex Casson - alex.casson@manchester.ac.uk
%
% Aim
% A LaTeX class file for submitting a BEng or MSc dissertation to the University of Manchester, particularly for courses in Electrical and Electronic Engineering
%
% Versions
% 28.03.25 - v5 - added option for BEng, and a wider number of accessibility changes to bring in sync with the PhD thesis template
% 25.09.23 - v4 - updated font and font size
% 13.07.21 - v3 - new version based upon thesis template to make this more of a class than an example
% 03.08.20 - v2 - initial version released on Overleaf by Farshad
% 01.06.20 - v1 - Beta testing only - never publicly released
% 
% Notes
% - Assumes build with XeLaTeX or LuaLaTeX. Font setting is different for LaTeX or PDFLaTeX
% - On Linux systems requires ttf-mscorefonts-installer to be installed to have access to the correct fonts
% - Based upon 2023 submission guidelines (https://documents.manchester.ac.uk/DocuInfo.aspx?DocID=2863)
% - Assumes title page is only one page long!
% - Assumes not using subfigure. Tocloft has (easily fixed) issues if do. 
% - Note that for the references font size it is set here to be one larger than that actually wanted. Not clear why, but then gives the correct size! Have to do this in main .tex
%
% To do
% - Caption size is fixed at 10pt for all fonts rather than changing with each font
% -----------------------------------------------------------------------------


%% Class definition - based upon report class
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{uom_eee_dissertation_casson}[2025/03/28 - Alex Casson - University of Manchester MSc and BEng dissertation format]
\LoadClass[a4paper, oneside]{article}


%% Load packages
\RequirePackage[a-3u]{pdfx}
%\RequirePackage{hyperref}                % create internal hyperlinks, must be specified before cite package to work. Is loaded automatically by pdfx
  \hypersetup{hidelinks,colorlinks=false,breaklinks=true,bookmarksdepth=3}  % have no highlighting on links and break lines in the LOF and LOT
\RequirePackage[explicit]{titlesec}      % re-format section titles
\RequirePackage{geometry}                % set paper margins
\RequirePackage{setspace}                % set line spacing
\RequirePackage[document]{ragged2e}      % set to be left aligned
\RequirePackage{pgfcore}                 % for logo positioning in fixed location
\RequirePackage{tikz}                    % for logo positioning in fixed location
  \usetikzlibrary{calc}                  % to allow coordinate calculations
\RequirePackage{fontspec}                % for font setting
\RequirePackage{tocloft} % allows precise control over the table of contents and deals with subfigures properly
  \renewcommand{\listfigurename}{List of figures} 
  \renewcommand{\listtablename}{List of tables} 
  \renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}} % add dots 
  \renewcommand{\cftsubsecleader}{\cftdotfill{\cftdotsep}}
  \setlength{\cftfignumwidth}{3em}  % modify number width in LoF
  \setlength{\cfttabnumwidth}{3em}  % modify number width in LoT
  \setlength{\cftfigindent}{0em}    % no indents in LoF
  \setlength{\cfttabindent}{0em}    % no indents in LoT
\RequirePackage{bookmark}                % for adding custom bookmarks
  \bookmarksetup{numbered}
\RequirePackage{enumitem}                % allow change of numbering style in enumeration
\RequirePackage{ifthen}                  % allow if then else statements
\RequirePackage{appendix}                % control over appendix
\RequirePackage{caption}                 % control figure captions
\RequirePackage{catchfile}               % used for counting the words and adding to a variable
\RequirePackage{extsizes}                % used to get the headings changing size with the passed in sizes
\RequirePackage[english=british]{csquotes} % used for adding quotes
  \renewcommand{\mkbegdispquote}[2]{\openautoquote\itshape} % adds quotation marks to displayed quotes and put in italics
  \renewcommand{\mkenddispquote}[2]{\closeautoquote#1#2} % adds quotation marks to displayed quotes
  \renewcommand{\mktextquote}[6]{#1\itshape#2#4#3#6#5} % put in-line quotes in italics
\RequirePackage[realmainfile]{currfile}  % used to get the LaTeX filename for the wordcount function
\RequirePackage{listings}                % used for environments to display code



%% Handle options that the class takes - font/size and meng vs. beng

\newcommand{\reportyear}{msc} % set report type
\setmainfont[Ligatures=TeX]{Arial}
\DeclareOption{msc}{
  \renewcommand{\reportyear}{msc}
  \setmainfont[Ligatures=TeX]{Arial}
}
\DeclareOption{beng}{
  \renewcommand{\reportyear}{beng}
  \setmainfont[Ligatures=TeX]{Carlito}
}

% Run the options
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{extreport}}
\ProcessOptions\relax


%% Set lengths - sizes for A4 paper from http://www.brand.manchester.ac.uk/visual-identity/print/
\newlength{\logowidth}    \setlength{\logowidth}{40.006mm}
\newlength{\logoheight}   \setlength{\logoheight}{19.454mm}
\newlength{\logopad}      \setlength{\logopad}{14.279mm}

\newlength{\marginleft}   \setlength{\marginleft}{2cm}   
\newlength{\marginright}  \setlength{\marginright}{2cm} 
\newlength{\margintop}    \setlength{\margintop}{2cm}
\newlength{\marginbottom} \setlength{\marginbottom}{2cm}%

\geometry{a4paper, top=\margintop, left=\marginleft, right=\marginright, bottom=\marginbottom, nohead, nofoot, nomarginpar}



%% Formatting options - mainly for font size
\setlength{\parskip}{2ex plus 0.5ex minus 0.2ex}
\setlength{\parindent}{0pt}
\linespread{1.5}
\pagestyle{plain} % simple page numbering
\setcounter{tocdepth}{3}                % set toc depth. This is changed for the Appendix
\setcounter{secnumdepth}{3}             % set how many levels of subsection to display numbers for. Note that if chaning this, bookmarksdepth and tocdepth probably want to be changed to match 
\renewcommand{\topfraction}{0.85}       % depreciate use of full pages for images
\renewcommand{\textfraction}{0.1}
\renewcommand{\floatpagefraction}{0.75}
\setlength{\footskip}{26pt}



% Format captions
\DeclareCaptionFormat{myformat}{\small#1#2#3}
\captionsetup{justification=centering,labelformat=simple,labelsep=period,labelfont=bf,format=myformat}
\captionsetup[figure]{justification=centering,labelformat=simple,labelsep=period,labelfont=bf,name={Fig.},format=myformat}

 

%% Format titles
\titleformat{\chapter}{\bfseries\Huge\raggedright}{\thechapter}{1ex}{#1} % default formatting for non-theses
\titlespacing{\chapter}{0pt}{-\baselineskip}{\baselineskip}
\ifthenelse{\equal{\reportyear}{thesis}}{\titleformat{\chapter}[display]{\bfseries\Huge\raggedright}{\chaptername~\thechapter}{0.5ex}{#1}}{}
\renewcommand{\contentsname}{\bfseries\Large Contents}
\renewcommand{\listfigurename}{\bfseries\Large List of figures}
\renewcommand{\listtablename}{\bfseries\Large List of tables}
\titleformat{\section}{\bfseries\Large}{\thesection}{1ex}{#1}
\titlespacing{\section}{0pt}{\baselineskip}{0.2\baselineskip}
\titleformat{\subsection}{\bfseries\large}{\thesubsection}{1ex}{#1}





%% Title page
\makeatletter
\def\faculty#1{\gdef\@faculty{#1}}
\def\school#1{\gdef\@school{#1}}
\def\departmentordivision#1{\gdef\@departmentordivision{#1}}
\def\course#1{\gdef\@course{#1}}
\def\studentid#1{\gdef\@studentid{#1}}
\def\submitdate#1{\gdef\@submitdate{#1}}
\newcommand{\titletext}[1]
{
  \ifthenelse{\equal{#1}{msc}}{A dissertation submitted to The University of Manchester for the degree of\ \linebreak \textbf{Master of Science in \@course}\ \linebreak in the Faculty of \@faculty}{}
  \ifthenelse{\equal{#1}{beng}}{A dissertation submitted to The University of Manchester for the degree of\ \linebreak \textbf{Bachelor of Engineering in \@course}\ \linebreak in the Faculty of \@faculty}{}
}
\newcommand{\submissiontext}{\textbf{Year of submission}\ \linebreak \@submitdate}
\newcommand{\authortext}{\textbf{Student ID}\ \linebreak \@studentid}
\def\maketitle{
  \begin{titlepage}
    \hypertarget{TitlePage}{}
    \bookmark[dest=TitlePage,level=-1]{Front matter}
	\phantomsection \pdfbookmark[section]{Title page}{title}
  
    % Add logo. No requirement for this, but is nice. Note sizes are hand calculated. Won't be correct if change paper size  
    \begin{tikzpicture}[overlay, remember picture] 
      \node[outer sep=0,inner sep=0,anchor=north west] at ($(current page.north west)+(\logopad,-\logopad)$){
        \includegraphics[alt={University of Manchester logo},width=\logowidth,keepaspectratio=true]{./images/uom_logo.pdf}
      };
    \end{tikzpicture}

    \begin{center}
      \vspace*{\fill}
      {\Large \bf \@title}
      \vskip 2cm
      \titletext{\reportyear}
      \vskip 2cm
      \submissiontext
      \vskip 2cm
      \authortext
      \vskip 2cm
      \@school
      \vspace*{\fill}
    \end{center}

  \end{titlepage}
  
  \pagenumbering{arabic}
  \setcounter{page}{2} % ensure no number on title page, numbers on other pages match those in the PDF
}
\makeatother  %to avoid error messages generated by "\@". Makes Latex treat "@" like a letter



%% Abstract
\renewenvironment{abstract}{%
  \clearpage\phantomsection\section*{Abstract}
}
{\addcontentsline{toc}{section}{\abstractname}\endquotation\vfil\null\clearpage}



%% Lists and tables of contents
\newcommand{\uomtoc}{
  \phantomsection\addcontentsline{toc}{section}{Contents}
  \tableofcontents
  \begin{flushright}
    \textbf{Word count}: \thewordcount
  \end{flushright}
  \clearpage
}

\newcommand{\uomlof}{
  \phantomsection\addcontentsline{toc}{section}{List of figures}
  \listoffigures
  \clearpage
  }
  
\newcommand{\uomlot}{
  \phantomsection\addcontentsline{toc}{section}{List of tables}
  \listoftables
  \clearpage
  }  
  
\newenvironment{uomlop}{
  \phantomsection\addcontentsline{toc}{section}{List of publications}
  \section*{List of publications}
}
{\clearpage}

\newenvironment{uomterms}{
  \phantomsection\addcontentsline{toc}{section}{Terms and abbreviations}
  \section*{Terms and abbreviations}
}
{\clearpage}

\newenvironment{uomlay}{
  \phantomsection\addcontentsline{toc}{section}{Lay abstract}
  \section*{Lay abstract}
}
{\clearpage}



%% Declaration of originality
\newcommand{\uomoriginalitydeclaration}{I hereby confirm that no portion of the work referred to in the thesis has been submitted in support of an application for another degree or qualification of this or any other university or other institute of learning.
}

\newenvironment{uomoriginality}{
  \phantomsection\addcontentsline{toc}{section}{Declaration of originality}
  \section*{Declaration of originality}
}
{\clearpage}

  

%% Copyright statement
\newcommand{\uomcopyrightstatement}{
  \phantomsection\addcontentsline{toc}{section}{Copyright statement}
  \section*{Copyright statement} 
  \begin{enumerate}[label=\roman*]
    \item The author of this thesis (including any appendices and/or schedules to this thesis) owns certain copyright or related rights in it (the ``Copyright'') and s/he has given The University of Manchester certain rights to use such Copyright, including for administrative purposes.
    \item Copies of this thesis, either in full or in extracts and whether in hard or electronic copy, may be made \emph{only} in accordance with the Copyright, Designs and Patents Act 1988 (as amended) and regulations issued under it or, where appropriate, in accordance with licensing agreements which the University has from time to time. This page must form part of any such copies made.
    \item The ownership of certain Copyright, patents, designs, trademarks and other intellectual property (the ``Intellectual Property'') and any reproductions of copyright works in the thesis, for example graphs and tables (``Reproductions''), which may be described in this thesis, may not be owned by the author and may be owned by third parties. Such Intellectual Property and Reproductions cannot and must not be made available for use without the prior written permission of the owner(s) of the relevant Intellectual Property and/or Reproductions.
	\item Further information on the conditions under which disclosure, publication and commercialisation of this thesis, the Copyright and any Intellectual Property and/or Reproductions described in it may take place is available in the University IP Policy (see \url{http://documents.manchester.ac.uk/DocuInfo.aspx?DocID=24420}), in any relevant Thesis restriction declarations deposited in the University Library, The University Library’s regulations (see \url{http://www.library.manchester.ac.uk/about/regulations/}) and in The University’s policy on Presentation of Theses.
  \end{enumerate}
  \clearpage
}



%% Acknowledgements  
\newenvironment{uomacknowledgements}{
  \phantomsection\addcontentsline{toc}{section}{Acknowledgments}
  \section*{Acknowledgments}
}
{\clearpage}



%% The author  
\newenvironment{uomauthor}{
  \phantomsection\addcontentsline{toc}{section}{The author}
  \section*{The author}
}
{\clearpage}



%% New command to start the main body, is used to reset the PDF index and end the 'Front matter' grouping.
\newcommand{\uomstartmainbody}{\bookmarksetup{startatroot}}



%% Appendix
\newenvironment{uomappendix}{
  \clearpage
  \bookmarksetupnext{level=-1}
  \phantomsection\addcontentsline{toc}{section}{Appendices}
  \addtocontents{toc}{\protect\setcounter{tocdepth}{2}}
  \section*{Appendices} 
  \begin{appendices}
}
{\end{appendices}}


%% Set autoref style for cross-references
\renewcommand{\appendixautorefname}{Appendix}%
\renewcommand{\chapterautorefname}{Chapter}%
\renewcommand{\sectionautorefname}{Section}%
\renewcommand{\subsectionautorefname}{Section}%
\renewcommand{\figureautorefname}{Fig.}%
\renewcommand{\tableautorefname}{Table}%
\renewcommand{\lstlistingname}{Listing}%
\def\equationautorefname#1{}%for autoref, gobble the space
\labelformat{equation}{(#1)}%
% Template uses \autoref so people can add their own text with just \ref if wanted, but could change this by uncommenting the below and then just use \ref everywhere. Note the \labelformat{equation} behaves differently and is used above
%\labelformat{appendix}{Appendix~#1}%
%\labelformat{chapter}{Chapter~#1}%
%\labelformat{section}{Section~#1}%
%\labelformat{subsection}{Section~#1}%
%\labelformat{figure}{Fig.~#1}%
%\labelformat{table}{Table~#1}%
%\labelformat{lstlisting}{Listing #1} % see https://tex.stackexchange.com/questions/655971/how-to-use-labelformat-for-lstlisting-with-listings-package-so-that-vref-of as may need additional code to work



%% Word count command and automatic counting function

% Word count command
\makeatletter
\newcommand{\wordcount}[1]{\newcommand{\@wordcount}{#1}}
\makeatother
\makeatletter\newcommand{\thewordcount}[0]{\@wordcount}\makeatother

% See https://www.overleaf.com/learn/how-to/Is_there_a_way_to_run_a_word_count_that_doesn%27t_include_LaTeX_commands%3F for info on what this counts.
\newcommand{\quickwordcount}[1]{%
  \CatchFileEdef\mywordcount{|"texcount -1 -sum -merge #1.tex"}{\endlinechar=-1 }%
}



%% Set up listings environment for displaying code
\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{1,1,1}
\lstdefinestyle{uom_code_style}{
  backgroundcolor=\color{backcolour},   
  commentstyle=\color{codegreen},
  keywordstyle=\color{magenta},
  numberstyle=\small\color{codegray},
  stringstyle=\color{codepurple},
  basicstyle=\ttfamily,
  breakatwhitespace=false,         
  breaklines=true,                 
  captionpos=t,                    
  keepspaces=true,                 
  numbers=left,                    
  numbersep=5pt,                  
  showspaces=false,                
  showstringspaces=false,
  showtabs=false,                  
  tabsize=2,
  xleftmargin=1em
}
\lstset{style=uom_code_style}