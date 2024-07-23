*** Settings ***
Documentation     Executes Google image search and stores the first result image.
Library           RPA.Browser.Selenium

*** Variables ***
${GOOGLE_URL}     https://google.com/?hl=en
${SEARCH_TERM}    cute cat picture

*** Keywords ***
Reject Google Cookies
    Click Element If Visible    xpath://button/div[contains(text(), 'Reject all')]

Accept Google Consent
    Click Element If Visible    xpath://button/div[contains(text(), 'I agree')]

Close Google Sign in if shown
    Click Element If Visible    No thanks

*** Keywords ***
Open Google search page
    Open Available Browser
    ...    ${GOOGLE_URL}
    ...    browser_selection=firefox
    ...    headless=True
    Close Google Sign in if shown
    Reject Google Cookies
    Accept Google Consent

*** Keywords ***
Search for
    [Arguments]    ${text}
    Input Text    name:q    ${text}
    Press Keys    name:q    ENTER
    Wait Until Page Contains Element    search

*** Keywords ***
View image search results
    Click Link    Images

*** Keywords ***
Screenshot first result
    Capture Element Screenshot    css:div[data-ri="0"]

*** Tasks ***
Execute Google image search and store the first result image
    TRY
        Open Google search page
        Search for    ${SEARCH_TERM}
        View image search results
        Screenshot first result
    EXCEPT
        Capture Page Screenshot     %{ROBOT_ARTIFACTS}${/}error.png 
        Fail    Checkout the screenshot: error.png
    END
    [Teardown]    Close Browser