"use client"
import { redirect } from "next/navigation";

export default function RootPage() {
  // return redirect("auth/login");
  if (localStorage.getItem("loginrole")){
  let role = localStorage.getItem("loginrole");
  if (role) {
    if (role === "officer") {
    return redirect("/dashboardofficer/homepage");
  } else if (role === "admin") {
    return redirect("/homead");
  }
  }}
  return redirect("/homeguest");
}
