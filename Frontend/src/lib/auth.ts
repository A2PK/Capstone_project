"use client"
export function getRole() {
if (localStorage.getItem("loginrole")){
    let role = localStorage.getItem("loginrole");
    if ((role == "admin") || (role == "officer")) return role;
}
return "";
  }