# Настройка прозрачной авторизации (SSO) для Web-панели

Для того чтобы реализовать **прозрачную авторизацию** (Single Sign-On / SSO) пользователей в веб-панели администратора (чтобы они попадали в Справочник без пароля), используется проксирование через веб-сервер с поддержкой NTLM/Kerberos.

## Архитектура решения

1. **Веб-сервер (IIS на Windows или Nginx на Linux)** выступает в роли обратного прокси-сервера (Reverse Proxy).
2. **Встроенная проверка подлинности Windows (Windows Authentication)** включается на уровне веб-сервера.
3. Веб-сервер перехватывает NTLM/Kerberos токен пользователя компьютера, состоящего в домене AD.
4. Веб-сервер передает имя авторизованного пользователя (например, `DOMAIN\ivanov`) в бэкенд FastAPI через защищенный HTTP-заголовок `X-Remote-User`.
5. **Приложение FastAPI (`admin/app.py`)** читает этот заголовок. Если заголовок присутствует, оно само авторизует пользователя и автоматически выдает ему роль `viewer` (Только чтение) для телефонного справочника, если его еще нет в БД.

---

## Настройка IIS на Windows Server (Рекомендуется)

### Шаг 1. Настройка IIS
1. Установите роль веб-сервера IIS вместе с компонентом **Windows Authentication** (Проверка подлинности Windows).
2. Установите модуль **URL Rewrite** и **Application Request Routing (ARR)** для IIS (бесплатные расширения от Microsoft).
3. Создайте новый сайт в IIS.
4. В разделе **Проверка подлинности (Authentication)**:
   - Отключите Анонимную проверку подлинности (Anonymous Authentication).
   - Включите Проверку подлинности Windows (Windows Authentication).

### Шаг 2. Настройка проксирования в `web.config`
В корневой папке вашего сайта создайте файл `web.config`, который будет проксировать запросы на локальный порт FastAPI (например, `8000`), а также передавать логин пользователя:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <system.webServer>
        <rewrite>
            <rules>
                <rule name="ReverseProxyToFastAPI" stopProcessing="true">
                    <match url="(.*)" />
                    <action type="Rewrite" url="http://127.0.0.1:8000/{R:1}" />
                    <serverVariables>
                        <!-- Передаем имя пользователя AD (LOGON_USER) в заголовок X-Remote-User -->
                        <set name="HTTP_X_REMOTE_USER" value="{LOGON_USER}" />
                    </serverVariables>
                </rule>
            </rules>
        </rewrite>
    </system.webServer>
</configuration>
```

*(Примечание: в IIS на уровне всего сервера нужно разрешить изменение серверной переменной `HTTP_X_REMOTE_USER` в настройках URL Rewrite -> View Server Variables).*

---

## Настройка на сервере Ubuntu Linux (Через Apache2 + GSSAPI)

Если вы разворачиваете проект на Ubuntu/Debian, самым надежным и простым способом включить прозрачную авторизацию (Kerberos/SSO) является использование веб-сервера **Apache2** в качестве обратного прокси с модулем `mod_auth_gssapi`.

### Шаг 1. Подготовка Keytab-файла на контроллере домена (Windows Server)
На контроллере домена (DC) с помощью утилиты `ktpass` нужно создать пользователя и сгенерировать keytab-файл для вашего Linux-сервера. 
Пример команды на контроллере домена:
```cmd
ktpass -princ HTTP/tempo-spravochnik.domain.loc@DOMAIN.LOC -mapuser web_sso_user -pass YourStrongPassword -crypto All -ptype KRB5_NT_PRINCIPAL -out http.keytab
```
Перенесите полученный файл `http.keytab` на сервер Ubuntu (например, в `/etc/apache2/http.keytab`) и дайте права веб-серверу:
```bash
sudo chown www-data:www-data /etc/apache2/http.keytab
sudo chmod 600 /etc/apache2/http.keytab
```

### Шаг 2. Установка пакетов на Ubuntu
```bash
sudo apt update
sudo apt install apache2 libapache2-mod-auth-gssapi krb5-user
```
*(При установке `krb5-user` введите имя вашего домена большими буквами, например: `DOMAIN.LOC`)*

### Шаг 3. Включение модулей проксирования
```bash
sudo a2enmod proxy proxy_http auth_gssapi headers rewrite
```

### Шаг 4. Настройка виртуального хоста (Virtual Host)
Создайте или отредактируйте конфигурационный файл сайта (например, `/etc/apache2/sites-available/itempo.conf`):

```apache
<VirtualHost *:80>
    ServerName tempo-spravochnik.domain.loc

    # Настройки Kerberos SSO
    <Location />
        AuthType GSSAPI
        AuthName "iTEMPO Directory SSO"
        GssapiCredStore keytab:/etc/apache2/http.keytab
        # Разрешить Kerberos (Negotiate)
        Require valid-user
        
        # Захватываем имя пользователя из Kerberos и кладем в HTTP-заголовок
        RequestHeader set X-Remote-User expr=%{REMOTE_USER}
        
        # Проксируем запросы в Python (FastAPI)
        ProxyPass http://127.0.0.1:8000/
        ProxyPassReverse http://127.0.0.1:8000/
    </Location>
</VirtualHost>
```

### Шаг 5. Применение настроек
```bash
sudo a2ensite itempo.conf
sudo systemctl restart apache2
```

Теперь пользователи с рабочих компьютеров Windows при переходе на `http://tempo-spravochnik.domain.loc` будут прозрачно авторизоваться через Apache, а FastAPI получит заголовок `X-Remote-User: ivanov@DOMAIN.LOC` (или `DOMAIN\ivanov`) и автоматически впустит в систему.

---

## Итог

Со стороны кода веб-админки всё уже готово. Как только системный администратор настроит IIS по инструкции выше, пользователи смогут заходить по ссылке в браузере и мгновенно попадать в телефонный справочник без ввода пароля.
